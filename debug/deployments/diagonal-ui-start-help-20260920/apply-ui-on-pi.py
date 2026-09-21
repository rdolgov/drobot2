"""Replace reviewed static UI files without restarting or commanding the robot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--sha256", required=True)
args = parser.parse_args()
repo = Path("/home/rd/drobot2").resolve(strict=True)
prefix = "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/four_leg_static/"
allowed = {prefix + "app.js", prefix + "index.html"}
digest = lambda data: hashlib.sha256(data).hexdigest()
if digest(args.archive.read_bytes()) != args.sha256:
    raise RuntimeError("Uploaded archive checksum differs")
with zipfile.ZipFile(args.archive) as archive:
    manifest = json.loads(archive.read("manifest.json"))
    if {entry["path"] for entry in manifest["files"]} != allowed:
        raise RuntimeError("Unexpected static file list")
    if set(archive.namelist()) != allowed | {"manifest.json"}:
        raise RuntimeError("Unexpected archive entries")
    changes = []
    for entry in manifest["files"]:
        target = (repo / entry["path"]).resolve(strict=True)
        if not target.is_relative_to(repo):
            raise RuntimeError("Static path escapes repository")
        old, new = target.read_bytes(), archive.read(entry["path"])
        if digest(old) not in (entry["old_sha256"], entry["new_sha256"]):
            raise RuntimeError("Pi static file changed: " + entry["path"])
        if digest(new) != entry["new_sha256"]:
            raise RuntimeError("Candidate checksum differs")
        changes.append((target, entry, old, new))

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
backup = repo / "debug/deployment-backups" / ("diagonal-ui-start-help-" + stamp)
backup.mkdir(parents=True, exist_ok=False)
for target, entry, old, new in changes:
    saved = backup / entry["path"]
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(old)
(backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

def replace(target, data):
    temporary = target.with_name(target.name + "." + stamp + ".ui-new")
    with temporary.open("xb") as stream:
        stream.write(data)
    shutil.copymode(target, temporary)
    os.replace(temporary, target)

written = []
try:
    for target, entry, old, new in changes:
        replace(target, new)
        written.append((target, old))
except BaseException:
    for target, old in reversed(written):
        replace(target, old)
    raise
result = {"installed": True, "backup": str(backup), "service_restarted": False,
          "controller_changed": False, "motion_commanded": False,
          "tests_run": False, "ui_validation_run": False}
(backup / "applied.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
