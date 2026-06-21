"""
NetShield AI-IDS — ML Engine
Random Forest + Anomaly Detection + Rule Engine
"""
import numpy as np
import json, time, random, hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque


# ── DATA STRUCTURES ──────────────────────────────────────────
@dataclass
class NetworkPacket:
    packet_id: str
    timestamp: float
    src_ip: str; dst_ip: str
    src_port: int; dst_port: int
    protocol: str; duration: float
    bytes_sent: int; bytes_received: int
    packets_sent: int; packets_received: int
    flag_syn: int; flag_ack: int; flag_fin: int; flag_rst: int; flag_urg: int
    service: str; connection_state: str

@dataclass
class ThreatEvent:
    event_id: str; timestamp: str
    src_ip: str; dst_ip: str; dst_port: int; protocol: str
    attack_type: str; severity: str
    confidence: float; blocked: bool
    bytes_transferred: int
    rule_triggered: str; mitre_tactic: str; mitre_technique: str
    response_action: str; anomaly_score: float; raw_score: float


# ── FEATURE EXTRACTOR ─────────────────────────────────────────
class FeatureExtractor:
    PROTO = {"TCP":0,"UDP":1,"ICMP":2,"HTTP":3,"HTTPS":4,"FTP":5,"SSH":6,"DNS":7}
    SVC   = {"http":0,"https":1,"ftp":2,"ssh":3,"smtp":4,"dns":5,"other":6}
    STATE = {"ESTABLISHED":0,"SYN_SENT":1,"SYN_RECV":2,"FIN_WAIT":3,"TIME_WAIT":4,"CLOSE":5}
    PRIV_PORTS      = set(range(1,1024))
    SENSITIVE_PORTS = {22,23,3306,5432,6379,27017,8080,8443,9200,2181}

    def extract(self, p: NetworkPacket) -> np.ndarray:
        br  = p.bytes_sent / max(p.bytes_received,1)
        pr  = p.packets_sent / max(p.packets_received,1)
        fc  = p.flag_syn+p.flag_ack+p.flag_fin+p.flag_rst+p.flag_urg
        syn = 1 if p.flag_syn==1 and p.flag_ack==0 else 0
        return np.array([
            self.PROTO.get(p.protocol,8), self.SVC.get(p.service,6),
            self.STATE.get(p.connection_state,6),
            min(p.duration,1000), min(p.bytes_sent,1e6), min(p.bytes_received,1e6),
            min(p.packets_sent,10000), min(p.packets_received,10000),
            p.flag_syn,p.flag_ack,p.flag_fin,p.flag_rst,p.flag_urg,
            br, pr,
            1 if p.src_port in self.PRIV_PORTS else 0,
            1 if p.dst_port in self.PRIV_PORTS else 0,
            1 if p.dst_port in self.SENSITIVE_PORTS else 0,
            fc, syn,
            1 if p.flag_rst==1 or p.flag_fin==1 else 0,
            1 if p.bytes_sent>100_000 else 0,
            1 if p.packets_sent>500 else 0,
            np.log1p(p.duration), np.log1p(p.bytes_sent+p.bytes_received),
        ], dtype=np.float32)


# ── RANDOM FOREST (pure numpy) ────────────────────────────────
class _Node:
    def __init__(self,fi,th,l=None,r=None,pred=None):
        self.fi=fi;self.th=th;self.l=l;self.r=r;self.pred=pred
    def predict(self,x):
        if self.pred is not None: return self.pred
        return self.l.predict(x) if x[self.fi]<=self.th else self.r.predict(x)

def _leaf(c): return _Node(0,0,pred=c)

class RandomForestIDS:
    CLASSES  = ["Normal","Port Scan","Brute Force","DoS/DDoS",
                 "SQL Injection","Malware C2","Data Exfiltration","Privilege Escalation"]
    SEV      = {0:"LOW",1:"MEDIUM",2:"HIGH",3:"CRITICAL",4:"HIGH",5:"CRITICAL",6:"HIGH",7:"CRITICAL"}
    MITRE_T  = {0:("—","—"),1:("Reconnaissance","T1046"),2:("Credential Access","T1110"),
                3:("Impact","T1499"),4:("Initial Access","T1190"),5:("C2","T1071"),
                6:("Exfiltration","T1041"),7:("Priv Escalation","T1068")}
    RESPONSE = {0:"ALLOW",1:"ALERT",2:"BLOCK_IP",3:"RATE_LIMIT",4:"BLOCK_IP",5:"QUARANTINE",6:"BLOCK_IP",7:"QUARANTINE"}

    def __init__(self):
        self.trees = self._build()

    def _build(self):
        trees=[]
        # Port scan
        trees.append(_Node(19,0.5,_leaf(0),_Node(22,0.5,_Node(3,0.1,_leaf(1),_leaf(0)),_leaf(1))))
        # DDoS
        trees.append(_Node(22,0.5,_leaf(0),_Node(21,0.5,_Node(8,0.5,_leaf(0),_leaf(3)),_leaf(3))))
        # Brute force
        trees.append(_Node(17,0.5,_leaf(0),_Node(6,100,_leaf(0),_Node(4,5000,_leaf(2),_leaf(0)))))
        # Exfiltration
        trees.append(_Node(21,0.5,_leaf(0),_Node(23,3.0,_leaf(0),_Node(13,5.0,_leaf(0),_leaf(6)))))
        # Malware C2
        trees.append(_Node(0,1.5,_Node(22,0.5,_leaf(0),_Node(3,300,_leaf(0),_leaf(5))),_leaf(0)))
        # SQL Injection
        trees.append(_Node(1,0.5,_leaf(0),_Node(4,2000,_leaf(0),_Node(24,7.0,_leaf(4),_leaf(4)))))
        # Priv esc
        trees.append(_Node(11,0.5,_leaf(0),_Node(17,0.5,_leaf(0),_Node(9,0.5,_leaf(7),_leaf(0)))))
        for d in [0.1,0.2,-0.1,0.15,-0.05,0.25,-0.15,0.05]:
            trees.append(_Node(19,0.5,
                _Node(22,0.5+d,_Node(21,0.5,_leaf(0),_Node(17,0.5,_leaf(6 if d>0 else 0),_leaf(3))),
                      _Node(17,0.5,_leaf(2),_leaf(5 if d<0 else 3))),
                _Node(3,0.5+d*2,_leaf(1),_Node(21,0.5,_leaf(0),_leaf(6)))))
        return trees

    def predict(self, feat):
        v=np.zeros(len(self.CLASSES))
        for t in self.trees: v[t.predict(feat)]+=1
        probs=v/len(self.trees); cls=int(np.argmax(probs))
        return cls, float(probs[cls]), probs

    def classify(self, pkt, feat):
        cls,conf,probs = self.predict(feat)
        mt,mte = self.MITRE_T[cls]
        return {
            "cls":cls,"attack_type":self.CLASSES[cls],"severity":self.SEV[cls],
            "confidence":round(conf,4),"mitre_tactic":mt,"mitre_technique":mte,
            "response_action":self.RESPONSE[cls],
            "blocked":self.RESPONSE[cls] in ("BLOCK_IP","QUARANTINE","RATE_LIMIT"),
            "class_probs":{self.CLASSES[i]:round(float(p),4) for i,p in enumerate(probs)},
        }


# ── ANOMALY DETECTOR ──────────────────────────────────────────
class AnomalyDetector:
    def __init__(self, w=200):
        self.history=deque(maxlen=w); self.mean=None; self.std=None
    def update(self, f):
        self.history.append(f)
        if len(self.history)>=10:
            a=np.array(self.history)
            self.mean=a.mean(0); self.std=a.std(0)+1e-8
    def score(self, f):
        if self.mean is None: return 0.0
        return float(np.clip(np.abs((f-self.mean)/self.std).mean()/5,0,1))


# ── RULE ENGINE ───────────────────────────────────────────────
class RuleEngine:
    RULES=[
        ("R001","SYN Flood",        lambda f: f[19]>0.5 and f[22]>0.5),
        ("R002","NULL Scan",         lambda f: f[18]==0 and f[6]>10),
        ("R003","Xmas Tree",         lambda f: f[8]>0.5 and f[10]>0.5 and f[12]>0.5),
        ("R004","Data Dump",         lambda f: f[21]>0.5 and f[13]>10),
        ("R005","Credential Brute",  lambda f: f[17]>0.5 and f[6]>100 and f[4]<5000),
        ("R006","C2 Beacon",         lambda f: f[3]>300 and f[4]<500 and f[6]<20),
        ("R007","Port Probe",        lambda f: f[19]>0.5 and f[3]<0.1),
        ("R008","RST Injection",     lambda f: f[11]>0.5 and f[17]>0.5),
    ]
    def match(self, f):
        for rid,name,fn in self.RULES:
            try:
                if fn(f): return f"{rid}: {name}"
            except: pass
        return "No rule match"


# ── PACKET SIMULATOR ──────────────────────────────────────────
class PacketSimulator:
    SUBNETS  = ["192.168.1.","10.0.0.","172.16.0.","203.45.67.","89.234.12.","45.33.32.","198.51.100."]
    PROTOCOLS= ["TCP","UDP","ICMP","HTTP","HTTPS","FTP","SSH","DNS"]
    SERVICES = ["http","https","ftp","ssh","smtp","dns","other"]
    STATES   = ["ESTABLISHED","SYN_SENT","SYN_RECV","FIN_WAIT","TIME_WAIT","CLOSE"]
    PROFILES = {
        "port_scan":    {"dur":(0.001,0.05),"bs":(40,100),"br":(0,60),"ps":(1,3),"syn":1,"ack":0,"ports":list(range(1,1025)),"w":0.12},
        "brute_force":  {"dur":(0.1,2.0),"bs":(200,2000),"br":(100,500),"ps":(50,500),"syn":1,"ack":1,"ports":[22,21,3389,23],"w":0.10},
        "ddos":         {"dur":(0.001,0.5),"bs":(60,1500),"br":(0,100),"ps":(500,5000),"syn":1,"ack":0,"ports":[80,443,8080],"w":0.08},
        "exfil":        {"dur":(60,3600),"bs":(100000,10000000),"br":(1000,10000),"ps":(100,1000),"syn":0,"ack":1,"ports":[443,80,53],"w":0.05},
        "malware":      {"dur":(300,86400),"bs":(100,2000),"br":(50,500),"ps":(5,30),"syn":0,"ack":1,"ports":[4444,8080,443,1337],"w":0.05},
        "normal":       {"dur":(0.1,30),"bs":(500,50000),"br":(500,100000),"ps":(5,100),"syn":1,"ack":1,"ports":[80,443,53,25,110],"w":0.60},
    }
    def _rip(self): return random.choice(self.SUBNETS)+str(random.randint(1,254))
    def _rv(self,r): return random.uniform(*r)
    def generate(self):
        names=list(self.PROFILES.keys()); total=sum(p["w"] for p in self.PROFILES.values())
        x=random.random(); acc=0; name=names[0]
        for n in names:
            acc+=self.PROFILES[n]["w"]/total
            if x<acc: name=n; break
        p=self.PROFILES[name]
        uid=hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
        return NetworkPacket(
            packet_id=uid, timestamp=time.time(),
            src_ip=self._rip(), dst_ip=self._rip(),
            src_port=random.randint(1024,65535),
            dst_port=random.choice(p["ports"]),
            protocol=random.choice(self.PROTOCOLS),
            duration=self._rv(p["dur"]),
            bytes_sent=int(self._rv(p["bs"])),
            bytes_received=int(self._rv(p["br"])),
            packets_sent=int(self._rv(p["ps"])),
            packets_received=random.randint(1,100),
            flag_syn=p.get("syn",random.randint(0,1)),
            flag_ack=p.get("ack",random.randint(0,1)),
            flag_fin=random.randint(0,1), flag_rst=random.randint(0,1), flag_urg=random.randint(0,1),
            service=random.choice(self.SERVICES),
            connection_state=random.choice(self.STATES),
        )


# ── IDS ENGINE (MAIN) ─────────────────────────────────────────
class IDSEngine:
    def __init__(self):
        self.extractor  = FeatureExtractor()
        self.model      = RandomForestIDS()
        self.anomaly    = AnomalyDetector()
        self.rules      = RuleEngine()
        self.simulator  = PacketSimulator()
        self.events: List[ThreatEvent] = []
        self.stats = {
            "total_packets":0,"threats_detected":0,"blocked":0,
            "by_severity":{"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0},
            "by_type":{},"session_start":datetime.now().isoformat()
        }

    def process(self, pkt: NetworkPacket) -> Optional[ThreatEvent]:
        self.stats["total_packets"]+=1
        feat = self.extractor.extract(pkt)
        self.anomaly.update(feat)
        ascore = self.anomaly.score(feat)
        ml = self.model.classify(pkt, feat)
        rule = self.rules.match(feat)
        cls  = ml["cls"]; conf = ml["confidence"]
        if ascore>0.7 and cls==0:
            ml["attack_type"]="Anomalous Traffic"; ml["severity"]="MEDIUM"
            conf=ascore*0.8; cls=1
        if cls==0 and ascore<0.3: return None
        sev=ml["severity"]
        self.stats["threats_detected"]+=1
        self.stats["by_severity"][sev]=self.stats["by_severity"].get(sev,0)+1
        self.stats["by_type"][ml["attack_type"]]=self.stats["by_type"].get(ml["attack_type"],0)+1
        if ml["blocked"]: self.stats["blocked"]+=1
        ev=ThreatEvent(
            event_id=pkt.packet_id, timestamp=datetime.fromtimestamp(pkt.timestamp).isoformat(),
            src_ip=pkt.src_ip, dst_ip=pkt.dst_ip, dst_port=pkt.dst_port, protocol=pkt.protocol,
            attack_type=ml["attack_type"], severity=sev, confidence=round(conf,4),
            blocked=ml["blocked"], bytes_transferred=pkt.bytes_sent+pkt.bytes_received,
            rule_triggered=rule, mitre_tactic=ml["mitre_tactic"], mitre_technique=ml["mitre_technique"],
            response_action=ml["response_action"], anomaly_score=round(ascore,4),
            raw_score=round(conf*(1+ascore*0.3),4),
        )
        self.events.append(ev)
        if len(self.events)>500: self.events=self.events[-500:]
        return ev

    def batch(self, n=10):
        evs=[]
        for _ in range(n):
            ev=self.process(self.simulator.generate())
            if ev: evs.append(ev)
        return evs

    def get_events(self, n=100, severity=None):
        evs=self.events[-n:][::-1]
        if severity and severity!="ALL": evs=[e for e in evs if e.severity==severity]
        return [asdict(e) for e in evs]

    def get_stats(self): return {**self.stats,"event_count":len(self.events)}

    def intel(self):
        evs=self.events[-100:]
        if not evs: return {}
        sevs=[e.severity for e in evs]; types=[e.attack_type for e in evs]
        return {
            "sample_size":len(evs),
            "severity_dist":{s:sevs.count(s) for s in ["CRITICAL","HIGH","MEDIUM","LOW"]},
            "top_attacks":sorted(set(types),key=types.count,reverse=True)[:5],
            "top_sources":list({e.src_ip for e in evs})[:5],
            "block_rate":round(sum(1 for e in evs if e.blocked)/len(evs),3),
            "avg_confidence":round(sum(e.confidence for e in evs)/len(evs),3),
            "mitre_tactics":list({e.mitre_tactic for e in evs if e.mitre_tactic!="—"}),
        }

    def reset(self):
        self.__init__()
