"""User-authorized single-joint +3 degree check through the existing bus owner."""
import argparse, json, math, re, signal, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--id",type=int,choices=(5,6),required=True)
args=parser.parse_args()
base="http://127.0.0.1:8080"
with urllib.request.urlopen(base+"/app.js",timeout=2) as response:
    script=response.read().decode()
match=re.search(r'const clientVersion = "([^"]+)"',script)
if not match: raise RuntimeError("Cannot identify dashboard client version")
headers={"Content-Type":"application/json","X-Drobot-Client-Version":match.group(1)}
out=Path("debug/hardware-calibration/individual-joint-check-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir(parents=True,exist_ok=False)
log={"id":args.id,"offset_deg":3.0,"leg":2,"started_at":datetime.now(timezone.utc).isoformat(),"samples":[],"result":"not_started"}
motor_number={5:2,6:3}[args.id]
original=None
last_sent=None
def request(path,payload=None):
    req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=2) as response: return json.load(response)
def motor_state():
    state=request("/api/state")
    if state["runtime"]["mode"]!="hardware": raise RuntimeError("Dashboard is not in hardware mode")
    if state.get("fault"): raise RuntimeError("Dashboard fault: "+str(state["fault"]))
    if state["rl_policy"]["active"] or state["crawl"]["active"]: raise RuntimeError("Autonomous motion is active")
    motor=next(m for leg in state["legs"] if leg["number"]==2 for m in leg["motors"] if m["id"]==args.id)
    return motor
def sample(phase,guard=True):
    motor=motor_state()
    log["samples"].append({"monotonic_s":time.monotonic(),"phase":phase,**motor})
    if guard:
        if not motor["armed"] or not motor["torque_enabled"]: raise RuntimeError("Test motor is disarmed; aborting")
        if motor["temperature_c"]>=50: raise RuntimeError("Test motor is too warm")
        if not 11.5 <= motor["voltage_v"] <= 12.8: raise RuntimeError("Voltage is outside the check envelope")
        if motor["current_ma"]>500: raise RuntimeError("Unexpected current on the unloaded joint")
        if motor.get("possible_stall"): raise RuntimeError("Motor reports possible stall")
        if abs(motor["measured_deg"]-last_sent)>2.0: raise RuntimeError("Tracking error exceeded two degrees")
        if abs(motor["desired_deg"]-last_sent)>.05: raise RuntimeError("Another control changed the test target")
    return motor
def target(degrees):
    global last_sent
    request("/api/target",{"leg":2,"motor":motor_number,"degrees":degrees})
    last_sent=degrees
def ramp(begin,end,label):
    steps=12
    for i in range(1,steps+1):
        sample(label)
        target(begin+(end-begin)*i/steps)
        time.sleep(.12)
    for _ in range(10):
        time.sleep(.1)
        sample(label+"_hold")
def interrupted(signum,frame):
    raise RuntimeError("Check interrupted by signal "+str(signum))
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,interrupted)
try:
    initial=motor_state()
    original=float(initial["desired_deg"]) if initial.get("desired_deg") is not None else None
    if original is None: raise RuntimeError("Existing holding target is required")
    last_sent=original
    if not initial["min_deg"]<=original+3<=initial["max_deg"]: raise RuntimeError("Check exceeds joint limits")
    sample("baseline")
    log["initial"]=initial
    print("Starting ID",args.id,"from hold",round(original,3),"to",round(original+3,3),"and back in 3 seconds",flush=True)
    time.sleep(3)
    sample("baseline_ready")
    ramp(original,original+3,"outward")
    reached=sample("peak")
    ramp(original+3,original,"return")
    returned=sample("returned")
    log.update(result="completed",original_hold_deg=original,target_hold_deg=original+3,
               initial_measured_deg=initial["measured_deg"],peak_measured_deg=reached["measured_deg"],
               measured_displacement_deg=reached["measured_deg"]-initial["measured_deg"],
               peak_error_deg=reached["measured_deg"]-(original+3),
               returned_measured_deg=returned["measured_deg"],return_error_deg=returned["measured_deg"]-original,
               initial_raw=initial["raw_position"],peak_raw=reached["raw_position"],returned_raw=returned["raw_position"],
               max_current_ma=max(s["current_ma"] for s in log["samples"]),
               max_temperature_c=max(s["temperature_c"] for s in log["samples"]))
except BaseException as exc:
    log["result"]="aborted"
    log["error"]=str(exc)
    try:
        request("/api/disarm",{"leg":2,"motor":motor_number})
        log["abort_disarm"]="acknowledged"
    except Exception as disarm:
        log["abort_disarm"]="failed: "+str(disarm)
    print("ABORTED",log["error"],log.get("abort_disarm"),flush=True)
finally:
    log["ended_at"]=datetime.now(timezone.utc).isoformat()
    (out/"result.json").write_text(json.dumps(log,indent=2)+"\n")
    print(json.dumps({k:v for k,v in log.items() if k not in ("samples","initial")}),flush=True)
    print("Saved",out/"result.json",flush=True)

