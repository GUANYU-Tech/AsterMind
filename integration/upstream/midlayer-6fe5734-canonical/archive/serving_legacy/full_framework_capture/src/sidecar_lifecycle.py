"""Scoped installation and cleanup for the scheduler sidecar startup hook."""

import os
import site
from pathlib import Path
from typing import Optional


SIDECAR_PTH_NAME = "midlayer_sidecar.pth"


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sidecar_pth_payload(project_root: Optional[Path] = None) -> str:
    root = Path(project_root) if project_root is not None else default_project_root()
    patch_path = root / "src" / "serve_trace" / "_sidecar_patch.py"
    operator_path = root / "src" / "serve_trace" / "_operator_patch.py"
    return (
        f"{root}\n"
        "import os, runpy; "
        f"runpy.run_path('{patch_path}') "
        "if os.environ.get('CUDA_SIDECAR_PATH') else None\n"
        "import os, runpy; "
        f"runpy.run_path('{operator_path}') "
        "if os.environ.get('CUDA_OPERATOR_TRACE_PATH') else None\n"
    )


def install_sidecar_pth(project_root: Optional[Path] = None,
                        site_dir: Optional[Path] = None) -> Optional[Path]:
    if not os.environ.get("CUDA_SIDECAR_PATH"):
        return None

    root = Path(project_root) if project_root is not None else default_project_root()
    target_dir = Path(site_dir) if site_dir is not None else Path(site.getusersitepackages())
    target_dir.mkdir(parents=True, exist_ok=True)
    pth_path = target_dir / SIDECAR_PTH_NAME
    pth_path.write_text(sidecar_pth_payload(root))
    return pth_path


def cleanup_sidecar_pth(pth_path: Optional[Path] = None,
                        project_root: Optional[Path] = None,
                        site_dir: Optional[Path] = None) -> bool:
    root = Path(project_root) if project_root is not None else default_project_root()
    if pth_path is None:
        target_dir = Path(site_dir) if site_dir is not None else Path(site.getusersitepackages())
        pth = target_dir / SIDECAR_PTH_NAME
    else:
        pth = Path(pth_path)

    try:
        current = pth.read_text()
    except FileNotFoundError:
        return False

    if current != sidecar_pth_payload(root):
        return False

    pth.unlink()
    return True
