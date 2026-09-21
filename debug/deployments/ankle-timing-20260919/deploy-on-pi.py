import hashlib,json,os,shutil,subprocess,zipfile
from datetime import datetime,timezone
from pathlib import Path

repo=Path.home()/"drobot2"
archive=Path.home()/"ankle-timing-update-20260919.zip"
expected_archive="c41fb002d896cb817645957e53b8c5c647b95dfc06f464464fcb465eda01a9c4"
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_archive:
    raise RuntimeError("Uploaded deployment archive differs from local package")
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read("deployment-manifest.json"))
    allowed={"onboard/policy-runtime/src/drobot_policy_runtime/ankle_transfer.py",
             "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/four_leg_control.py"}
    if {v["path"] for v in manifest["files"]}!=allowed:
        raise RuntimeError("Unexpected deployment paths")
    changes=[]
    for entry in manifest["files"]:
        target=(repo/entry["path"]).resolve()
        if not target.is_relative_to(repo.resolve()):
            raise RuntimeError("Invalid deployment path")
        current=target.read_bytes()
        candidate=z.read(entry["path"])
        if hashlib.sha256(current).hexdigest()!=entry["expected_previous_sha256"]:
            raise RuntimeError("Pi source changed since download: "+entry["path"])
        if hashlib.sha256(candidate).hexdigest()!=entry["new_sha256"]:
            raise RuntimeError("Candidate source hash mismatch")
        changes.append((target,entry["path"],candidate))
stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup=repo/"debug/deployment-backups"/("ankle-timing-"+stamp)
for target,relative,_ in changes:
    saved=backup/relative
    saved.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(target,saved)
(backup/"deployment-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
subprocess.run(["sudo","-n","systemctl","stop","drobot-manual-web"],check=True)
try:
    for target,relative,candidate in changes:
        temp=target.with_name(target.name+".timing.tmp")
        temp.write_bytes(candidate)
        shutil.copymode(target,temp)
        os.replace(temp,target)
    subprocess.run(["sudo","-n","systemctl","start","drobot-manual-web"],check=True)
except BaseException:
    for target,relative,_ in changes:
        shutil.copy2(backup/relative,target)
    subprocess.run(["sudo","-n","systemctl","start","drobot-manual-web"],check=False)
    raise
result={"deployed_at_utc":datetime.now(timezone.utc).isoformat(),
        "backup":str(backup),"files":[p for _,p,_ in changes],
        "dashboard_start_command":"succeeded","tests_run":False,
        "hardware_motion_check_run":False,"calibration_files_changed":False}
audit=repo/"debug/deployments/ankle-timing-20260919"
audit.mkdir(parents=True,exist_ok=True)
(audit/"applied.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))

