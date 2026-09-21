"""Capture immutable inputs for the authorized training runs, without exercising them."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'simulation/reviews/ankle-v23-transfer/provenance'
OUT.mkdir(parents=True,exist_ok=True)
sources=[*Path(__file__).parent.glob('*.py'),
         *Path(__file__).parent.parent.joinpath('ankle_v21').glob('*.py'),
         *Path(__file__).parent.parent.joinpath('ankle_v2').glob('*.py'),
         ROOT/'cad/exports/urdf/quadruped_robot_v23_solid_sole.urdf',
         ROOT/'simulation/exports/isaac/ankle_v23_stride/assets.json',
         ROOT/'simulation/exports/isaac/ankle_v23_stride/active/robot.usdc',
         ROOT/'cad/reviews/ankle-v23-stride/mass-properties.json',
         ROOT/'cad/drobot_cad/parts/ankle_v23_existing_leg.py']
hashes={}
for path in sources:
    relative=path.relative_to(ROOT)
    dest=OUT/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest)
    hashes[relative.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
warm=ROOT/'simulation/isaac/models/ankle-v23-stride/seed2301-model_2250.pt'
hashes[warm.relative_to(ROOT).as_posix()]=hashlib.sha256(warm.read_bytes()).hexdigest()
(OUT/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('V23_PROVENANCE_SAVED',flush=True)
