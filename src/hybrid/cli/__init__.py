"""
hybrid.cli
================

Command-line entrypoints for common pipelines:
- Motion correction (mc_cli)
- Preprocessing (prep_cli)
- QC map generation (qc_cli)
- Cellpose-ready input preparation (cp_prep_cli)

Re-exports
----------
from hybrid.cli import (
    run_mc, mc_main,
    run_prep, prep_main,
    cp_prep_process_one, cp_prep_main,
    run_qc, qc_main,
    # and modules:
    mc_cli, prep_cli, cp_prep_cli, qc_cli,
)
"""

# short imports, e.g.:
#   from hybrid.cli import run_mc, run_prep
#   from hybrid.cli import mc_cli, prep_cli
from .mc_cli import run_mc as run_mc, main as mc_main
from .prep_cli import run_prep as run_prep, main as prep_main
from .cp_prep_cli import run_cp_prep as run_cp_prep, main as cp_prep_main
from .qc_cli import run_qc as run_qc, main as qc_main

# also expose the submodules themselves
import importlib as _importlib
mc_cli = _importlib.import_module(".mc_cli", __name__)
prep_cli = _importlib.import_module(".prep_cli", __name__)
cp_prep_cli = _importlib.import_module(".cp_prep_cli", __name__)
qc_cli = _importlib.import_module(".qc_cli", __name__)

__all__ = [
    # functions
    "run_mc", "mc_main",
    "run_prep", "prep_main",
    "run_cp_prep", "cp_prep_main",
    "run_qc", "qc_main",
    # modules
    "mc_cli", "prep_cli", "cp_prep_cli", "qc_cli",
]
