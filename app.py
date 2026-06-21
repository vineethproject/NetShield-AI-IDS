"""
NetShield AI-IDS — Flask REST API Server
Run: python app.py
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading, time, random
from ids_engine import IDSEngine

app    = Flask(__name__)
CORS(app)
engine = IDSEngine()
lock   = threading.Lock()

def _bg():
    while True:
        with lock:
            engine.batch(random.randint(1,4))
        time.sleep(random.uniform(0.8, 2.5))

threading.Thread(target=_bg, daemon=True).start()

@app.route("/api/status")
def status():
    with lock: s=engine.get_stats()
    return jsonify({"status":"online","stats":s})

@app.route("/api/events")
def events():
    n   = min(int(request.args.get("n",100)),300)
    sev = request.args.get("severity","ALL")
    with lock: evs=engine.get_events(n,sev)
    return jsonify({"events":evs,"total":len(evs)})

@app.route("/api/intel")
def intel():
    with lock: i=engine.intel(); s=engine.get_stats()
    return jsonify({"intel":i,"stats":s})

@app.route("/api/simulate", methods=["POST"])
def simulate():
    n = min(int((request.json or {}).get("count",10)),50)
    with lock: evs=engine.batch(n)
    return jsonify({"processed":n,"threats":len(evs)})

@app.route("/api/reset", methods=["POST"])
def reset():
    with lock: engine.reset()
    return jsonify({"status":"reset"})

if __name__ == "__main__":
    print("IDS API → http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
