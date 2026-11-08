"""Root launcher for mmFormer MyoPS380 evaluation."""
import sys
from pathlib import Path

# Ensure repo root is at head of sys.path
root = Path(__file__).resolve().parent
pkg_dir = str(root / "mmformer")
while pkg_dir in sys.path:
    sys.path.remove(pkg_dir)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from mmformer.evaluate_myops import main

if __name__ == "__main__":
    main()
