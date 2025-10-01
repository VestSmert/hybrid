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
    mc_run_one, mc_main,
    prep_process_one, prep_main,
    cp_prep_process_one, cp_prep_main,
    qc_process_one, qc_main,
    # and modules:
    mc_cli, prep_cli, cp_prep_cli, qc_cli,
)
"""

# short imports, e.g.:
#   from hybrid.cli import mc_run_one, prep_process_one
#   from hybrid.cli import mc_cli, prep_cli
from .mc_cli import run_one as mc_run_one, main as mc_main
from .prep_cli import process_one as prep_process_one, main as prep_main
from .cp_prep_cli import process_one as cp_prep_process_one, main as cp_prep_main
from .qc_cli import process_one as qc_process_one, main as qc_main

# also expose the submodules themselves
import importlib as _importlib
mc_cli = _importlib.import_module(".mc_cli", __name__)
prep_cli = _importlib.import_module(".prep_cli", __name__)
cp_prep_cli = _importlib.import_module(".cp_prep_cli", __name__)
qc_cli = _importlib.import_module(".qc_cli", __name__)

__all__ = [
    # functions
    "mc_run_one", "mc_main",
    "prep_process_one", "prep_main",
    "cp_prep_process_one", "cp_prep_main",
    "qc_process_one", "qc_main",
    # modules
    "mc_cli", "prep_cli", "cp_prep_cli", "qc_cli",
]
