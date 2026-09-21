import json, os, shutil, subprocess, zipfile, hashlib
from datetime import datetime, timezone
from pathlib import Path

records = json.loads(r'''[{"servo_id":5,"candidate_center_tick":2101,"sample_span_deg":0.087890625,"captured_at_utc":"2026-09-19T19:36:30.632771+00:00","raw_samples":[2101,2100,2101,2101,2101],"user_confirmed_pose":"Hip aligned: servo 6 pivot directly below servo 5 pivot with body level","source_file":"debug/hardware-calibration/front-right-alignment-20260919/hip-aligned-20260919T193631Z.json","source_sha256":"6896f00173a2028cdfa340a8e42295fd1a62f918b8724fcc29af55f8d91bd91b"},{"servo_id":6,"candidate_center_tick":2054,"sample_span_deg":0,"captured_at_utc":"2026-09-19T19:50:21.859886+00:00","raw_samples":[2054,2054,2054,2054,2054],"user_confirmed_pose":"Knee aligned: hip, knee, and ankle pivot centers form a straight line viewed from the robot right side","source_file":"debug/hardware-calibration/front-right-alignment-20260919/knee-aligned-20260919T195022Z.json","source_sha256":"b09756c5fd57e482207d6b7f59821dc08f020984fd1afde75ce32768079e2ac5"},{"servo_id":14,"candidate_center_tick":2971,"sample_span_deg":0,"captured_at_utc":"2026-09-19T19:51:32.855856+00:00","raw_samples":[2971,2971,2971,2971,2971],"user_confirmed_pose":"Ankle aligned: sole perpendicular to knee-to-ankle pivot line in side view","source_file":"debug/hardware-calibration/front-right-alignment-20260919/ankle-aligned-20260919T195133Z.json","source_sha256":"26e4bd0cff1b2d662c5c3c5dcafaf2eb5ce58c5390a667f09fc1fa46330f4c95"}]''')
root=Path("hardware/robot-runtime/servos")
paths={key:root/name for key,name in {
    "leg":"calibration-leg-2.json",
    "ankles":"calibration-ankles.json",
    "travel":"ankle-travel-observations.json"}.items()}
docs={key:json.loads(path.read_text(encoding="utf-8")) for key,path in paths.items()}
leg={v["id"]:v for v in docs["leg"]["motors"]}
ankles={v["id"]:v for v in docs["ankles"]["motors"]}
travel={v["id"]:v for v in docs["travel"]["motors"]}
expected={5:2162,6:2051,14:2982}
for record in records:
    sid=record["servo_id"]
    if record["sample_span_deg"]>1.0:
        raise RuntimeError("Alignment capture moved too much")
    motor=ankles[sid] if sid==14 else leg[sid]
    if motor["center_tick"]!=expected[sid]:
        raise RuntimeError("Saved center changed since alignment began: "+str(sid))
if ankles[14]["direction"]!=1 or travel[14]["direction"]!=1 or travel[14]["center_tick"]!=expected[14]:
    raise RuntimeError("Ankle direction or paired travel record changed")
now=datetime.now(timezone.utc)
stamp=now.isoformat()
for record in records:
    sid=record["servo_id"]
    doc=docs["ankles"] if sid==14 else docs["leg"]
    motor=ankles[sid] if sid==14 else leg[sid]
    old=motor["center_tick"]
    motor["center_tick"]=record["candidate_center_tick"]
    doc["updated_at_utc"]=stamp
    doc.setdefault("neutral_center_updates",[]).append({
        "id":sid,"previous_center_tick":old,"center_tick":motor["center_tick"],
        "captured_at_utc":record["captured_at_utc"],"applied_at_utc":stamp,
        "reference_pose":record["user_confirmed_pose"],
        "capture_source":record["source_file"],"capture_sha256":record["source_sha256"],
        "sample_ticks":record["raw_samples"],"torque_off_during_capture":True,
        "powered_motion_verification":False})
old_center=travel[14]["center_tick"]
new_center=ankles[14]["center_tick"]
tick_shift=((old_center-new_center+2048)%4096)-2048
angle_shift=tick_shift*360/4096*ankles[14]["direction"]
travel[14]["center_tick"]=new_center
for key in ("min_angle_deg","max_angle_deg","last_angle_deg"):
    travel[14][key]+=angle_shift
docs["travel"]["calibration_updated_at_utc"]=stamp
docs["travel"]["updated_at_utc"]=stamp
docs["travel"].setdefault("coordinate_updates",[]).append({
    "id":14,"previous_center_tick":old_center,"center_tick":new_center,
    "direction":1,"angle_offset_deg":angle_shift,"updated_at_utc":stamp,
    "method":"Translate existing angles to the recaptured neutral center; raw encoder endpoints and five-degree inward margin unchanged. No new travel capture."})
backup=root/"backups"/("before-front-right-alignment-"+now.strftime("%Y%m%dT%H%M%SZ"))
backup.mkdir(parents=True,exist_ok=False)
for path in paths.values():
    shutil.copy2(path,backup/path.name)
audit_dir=Path("debug/hardware-calibration/front-right-alignment-20260919")
audit_dir.mkdir(parents=True,exist_ok=True)
(audit_dir/"alignment-capture-summary.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
subprocess.run(["sudo","-n","systemctl","stop","drobot-manual-web"],check=True)
try:
    for key,path in paths.items():
        temp=path.with_name(path.name+".alignment.tmp")
        temp.write_text(json.dumps(docs[key],indent=2)+"\n",encoding="utf-8")
        os.replace(temp,path)
except BaseException:
    for path in paths.values():
        shutil.copy2(backup/path.name,path)
    raise
subprocess.run(["sudo","-n","systemctl","start","drobot-manual-web"],check=True)
result={"applied_at_utc":stamp,"backup":str(backup),
    "centers":{str(sid):leg[sid]["center_tick"] for sid in (4,5,6)},
    "ankle_14_center":new_center,
    "ankle_14_direction":1,
    "ankle_14_limits_deg":[travel[14]["min_angle_deg"]+5,travel[14]["max_angle_deg"]-5],
    "ankle_14_coordinate_offset_deg":angle_shift,
    "dashboard_start_command":"succeeded",
    "powered_motion_verification":"not_run"}
(audit_dir/"alignment-applied.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
package=Path.home()/"front-right-alignment-applied-20260919.zip"
with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as z:
    for path in paths.values():
        z.write(path,path.as_posix())
    for name in ("alignment-capture-summary.json","alignment-applied.json"):
        z.write(audit_dir/name,(audit_dir/name).as_posix())
print(json.dumps({**result,"package":str(package),"package_sha256":hashlib.sha256(package.read_bytes()).hexdigest()}))

