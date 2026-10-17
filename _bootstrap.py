"""Load the flat research package consistently from any checkout directory."""
import importlib.util
from pathlib import Path
import sys


def activate():
    name = "driving"
    root = Path(__file__).resolve().parent
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                    submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif Path(sys.modules[name].__file__).resolve().parent != root:
        raise ImportError("A different driving package is already loaded")
    return name
