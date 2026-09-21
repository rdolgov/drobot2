"""Install a reviewed source-only IK bundle; no walking or automated verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path)
parser.add_argument("--sha256")
parser.add_argument("--repo", type=Path, default=Path.home()/"drobot2")
parser.add_argument("--apply", action="store_true")
parser.add_argument("--restore", type=Path, help="Restore a backup produced by this installer")
args = parser.parse_args()
repo = args.repo.resolve(strict=True)
service = "drobot-manual-web"
allowed = {
    "hardware/robot-runtime/four-leg.toml",
    "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/ankle_ik_gait.py",
    "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/four_leg_control.py",
    "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/four_leg_static/app.js",
    "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/four_leg_static/index.html",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def destination(relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts:
        raise RuntimeError("Invalid package path")
    target = (repo / relative).resolve()
    if not target.is_relative_to(repo) or target == repo:
        raise RuntimeError("Package path escapes repository")
    return target


def replace(target, data):
    temporary = target.with_name(target.name+"."+stamp+".ik-new")
    with temporary.open("xb") as stream:
        stream.write(data)
    if target.exists():
        shutil.copymode(target, temporary)
    os.replace(temporary, target)


stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
if args.restore:
    backup = args.restore.resolve(strict=True)
    if not backup.is_relative_to(repo / "debug/deployment-backups"):
        raise RuntimeError("Restore path must be a repository deployment backup")
    state = json.loads((backup / "backup.json").read_text())
    if {e["path"] for e in state["files"]} != allowed:
        raise RuntimeError("Unexpected backup file list")
    for entry in state["files"]:
        target = destination(entry["path"])
        if not target.exists() or digest(target.read_bytes()) != entry["new_sha256"]:
            raise RuntimeError("Current source differs from installed package: "+entry["path"])
        if entry["existed"] and digest((backup / entry["path"]).read_bytes()) != entry["old_sha256"]:
            raise RuntimeError("Backup checksum differs")
    if not args.apply:
        print("Backup paths and hashes match. Use --apply with --restore to replace sources.")
        raise SystemExit(0)
    subprocess.run(["sudo", "-n", "systemctl", "stop", service], check=True)
    for entry in state["files"]:
        target = destination(entry["path"])
        if entry["existed"]:
            replace(target, (backup / entry["path"]).read_bytes())
        else:
            target.unlink()
    if state["service_was_active"]:
        subprocess.run(["sudo", "-n", "systemctl", "start", service], check=True)
    print(json.dumps({"restored": str(backup), "motion_commanded": False}))
    raise SystemExit(0)

if args.archive is None or args.sha256 is None:
    parser.error("--archive and --sha256 are required for installation")
if digest(args.archive.read_bytes()) != args.sha256:
    raise RuntimeError("Uploaded archive SHA-256 differs")
with zipfile.ZipFile(args.archive) as archive:
    manifest = json.loads(archive.read("deployment-manifest.json"))
    if {e["path"] for e in manifest["files"]} != allowed:
        raise RuntimeError("Unexpected package file list")
    if set(archive.namelist()) != allowed | {"deployment-manifest.json"}:
        raise RuntimeError("Unexpected archive entries")
    changes = []
    for entry in manifest["files"]:
        target = destination(entry["path"])
        content = archive.read(entry["path"])
        if digest(content) != entry["new_sha256"]:
            raise RuntimeError("Candidate checksum differs: "+entry["path"])
        old = target.read_bytes() if target.exists() else None
        old_hash = digest(old) if old is not None else None
        if old_hash != entry["new_sha256"] and old_hash not in entry["expected_previous_sha256"]:
            raise RuntimeError("Pi source changed; review it before replacing: "+entry["path"])
        changes.append((target, entry, content, old))
    for dependency in manifest["read_only_prerequisites"]:
        target = destination(dependency["path"])
        if not target.is_file() or digest(target.read_bytes()) not in dependency["sha256"]:
            raise RuntimeError("Pi calibration/runtime prerequisite differs; not replacing: "+dependency["path"])
if not args.apply:
    print(json.dumps({"package": manifest["package"], "file_hashes_match": True,
                      "installed": False, "next": "Use --apply when ready to replace sources and restart the dashboard."}))
    raise SystemExit(0)
if all(old is not None and digest(old) == entry["new_sha256"] for _, entry, _, old in changes):
    print("Package is already installed; no files or service changed.")
    raise SystemExit(0)
service_state = subprocess.run(["systemctl", "is-active", "--quiet", service], check=False).returncode
if service_state not in (0, 3):
    raise RuntimeError("Cannot determine dashboard service state")
backup = repo / "debug/deployment-backups" / ("ik-diagonal-"+stamp)
backup.mkdir(parents=True, exist_ok=False)
state = {"service_was_active": service_state == 0, "files": []}
for target, entry, content, old in changes:
    if old is not None:
        saved = backup / entry["path"]
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(old)
    state["files"].append({"path": entry["path"], "existed": old is not None,
                           "old_sha256": digest(old) if old is not None else None,
                           "new_sha256": entry["new_sha256"]})
(backup / "backup.json").write_text(json.dumps(state, indent=2)+"\n")
subprocess.run(["sudo", "-n", "systemctl", "stop", service], check=True)
written = []
try:
    for target, entry, content, old in changes:
        replace(target, content)
        written.append((target, old))
    if state["service_was_active"]:
        subprocess.run(["sudo", "-n", "systemctl", "start", service], check=True)
except BaseException:
    for target, old in reversed(written):
        if old is None:
            target.unlink()
        else:
            replace(target, old)
    if state["service_was_active"]:
        subprocess.run(["sudo", "-n", "systemctl", "start", service], check=False)
    raise
result = {"package": manifest["package"], "backup": str(backup), "installed": True,
          "service_restarted": state["service_was_active"], "tests_run": False,
          "hardware_motion_check_run": False, "walk_commanded": False}
(backup / "applied.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))
