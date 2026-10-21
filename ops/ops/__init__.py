"""cfensemble-ops — remote-training infrastructure for cf-ensemble.

Decoupled from the ensemble library (`cfensemble`): nothing here imports it, and
`cfensemble` never imports this. Cloud infra is scaffolding for running
experiments at scale, not part of the method itself. Adapted from ssl-lab/ops.

SkyPilot orchestrates; RunPod is the provider (a config value).
"""

from ops.gpu_runner import (
    GPU_SPECS,
    InfraConfig,
    build_skypilot_config,
    down,
    down_all,
    estimate_cost,
    keepalive_config,
    launch,
    print_dry_run,
    provision,
    status,
)
from ops.datasets import data_root, link_dataset
from ops.hardware import HardwareReport, detect_hardware, print_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "InfraConfig",
    "GPU_SPECS",
    "build_skypilot_config",
    "print_dry_run",
    "launch",
    "keepalive_config",
    "provision",
    "status",
    "down",
    "down_all",
    "estimate_cost",
    "detect_hardware",
    "print_report",
    "HardwareReport",
    "data_root",
    "link_dataset",
]
