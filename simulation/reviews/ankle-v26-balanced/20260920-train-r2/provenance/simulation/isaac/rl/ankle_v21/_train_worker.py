import sys
from pathlib import Path
import warp as wp
wp.config.enable_backward = False
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from isaaclab_rl.entrypoints import run_train_cli
import ankle_v21
if __name__ == '__main__':
    raise SystemExit(run_train_cli())
