"""Root launcher for mmFormer MyoPS380 and MyoPS++ training."""
import sys
from pathlib import Path

# Ensure repo root is at head of sys.path
root = Path(__file__).resolve().parent
pkg_dir = str(root / "mmformer")
while pkg_dir in sys.path:
    sys.path.remove(pkg_dir)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from mmformer.train_myops import build_parser, main, parse_args

if __name__ == "__main__":
    main()
