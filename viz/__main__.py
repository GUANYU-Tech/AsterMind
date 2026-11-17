"""Start with ``conda activate OS; python -B -m viz``."""
from pathlib import Path
import sys

def main():
    frontend = str(Path(__file__).resolve().parent)
    if frontend not in sys.path:
        sys.path.insert(0, frontend)
    from runtime_environment import activate
    activate()
    from web_server import main as serve
    serve()

if __name__ == '__main__':
    main()
