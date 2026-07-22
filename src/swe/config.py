"""User-level configuration management for SWE.

Supports a ``~/.swe/config.yml`` file with user defaults for FDSN
client, output directories, preferred backends, and other settings
that are tedious to repeat on every CLI invocation.

Configuration keys
------------------
fdsn_client : str
    Default FDSN data centre (e.g. ``"IRIS"``, ``"SCEDC"``).
output_dir : str
    Default base output directory.
picker_backend : str
    Default picker (``"seisbench"`` or ``"stalta"``).
picker_model : str
    Default SeisBench model name (e.g. ``"PhaseNet"``).
associator_backend : str
    Default associator (``"dbscan"``, ``"pyocto"``, or ``"gamma"``).
run_dir : str
    Default pipeline run directory (default ``.swe_runs``).
parallel : bool
    Whether to run independent pipeline steps in parallel by default.
log_level : str
    Default logging level (``"WARNING"``, ``"INFO"``, ``"DEBUG"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = ["SWEConfig", "get_config_path", "load_config"]

_DEFAULT_CONFIG_DIR = Path.home() / ".swe"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.yml"


@dataclass
class SWEConfig:
    """User-level SWE configuration.

    All fields have sensible defaults and can be overridden from
    ``~/.swe/config.yml``.
    """

    fdsn_client: str = "IRIS"
    output_dir: str = "."
    picker_backend: str = "seisbench"
    picker_model: str = "PhaseNet"
    associator_backend: str = "dbscan"
    run_dir: str = ".swe_runs"
    parallel: bool = False
    log_level: str = "WARNING"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SWEConfig:
        """Create a config from a dictionary, ignoring unknown keys.

        Parameters
        ----------
        data : dict
            Configuration key-value pairs.

        Returns
        -------
        SWEConfig
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def get_config_path() -> Path:
    """Return the path to the user config file.

    Returns
    -------
    Path
        ``~/.swe/config.yml``
    """
    return _DEFAULT_CONFIG_FILE


def load_config(
    *,
    config_path: Path | None = None,
) -> SWEConfig:
    """Load the user configuration from disk.

    If the config file does not exist, returns the default configuration.

    Parameters
    ----------
    config_path : Path, optional
        Override path to the config file (default ``~/.swe/config.yml``).

    Returns
    -------
    SWEConfig
        The loaded (or default) configuration.
    """
    path = config_path or _DEFAULT_CONFIG_FILE

    if not path.exists():
        logger.debug("No config file at %s, using defaults", path)
        return SWEConfig()

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse config %s: %s", path, exc)
        return SWEConfig()

    if not isinstance(raw, dict):
        logger.warning("Config file %s is not a mapping, using defaults", path)
        return SWEConfig()

    logger.debug("Loaded config from %s", path)
    return SWEConfig.from_dict(raw)


def save_config(
    config: SWEConfig,
    *,
    config_path: Path | None = None,
) -> Path:
    """Save the configuration to disk.

    Creates the ``~/.swe/`` directory if it does not exist.

    Parameters
    ----------
    config : SWEConfig
        Configuration to save.
    config_path : Path, optional
        Override path (default ``~/.swe/config.yml``).

    Returns
    -------
    Path
        The path where the config was saved.
    """
    path = config_path or _DEFAULT_CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict

    data = asdict(config)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=True))
    logger.info("Saved config to %s", path)
    return path
