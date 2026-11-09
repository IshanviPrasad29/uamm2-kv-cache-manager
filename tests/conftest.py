import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_verilator():
    """Return a verilator executable path or None."""
    env = os.environ.get("VERILATOR")
    if env and shutil.which(env):
        return shutil.which(env)
    found = shutil.which("verilator")
    if found:
        return found
    try:                                    # pip-installed `verilator` package
        import verilator
        root = Path(verilator.__file__).parent
        cand = root / "bin" / "verilator"
        if cand.exists():
            os.environ.setdefault("VERILATOR_ROOT", str(root))
            return str(cand)
    except ImportError:
        pass
    return None
