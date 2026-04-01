"""config_utils.py — YAML config loader for train.py / test.py.

Usage pattern
─────────────
  parser = argparse.ArgumentParser()
  parser.add_argument('--foo', ...)
  ...
  args = load_config_and_parse(parser)

  # CLI always overrides YAML:
  #   python train.py --config configs/highd_baseline.yaml --lr 0.001

load_config_and_parse():
  1. Peeks at sys.argv for --config <path>.
  2. Loads the YAML and feeds it as argparse defaults.
  3. Parses the full CLI (explicit CLI values override YAML).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_config_and_parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add --config support to an existing ArgumentParser and return parsed args.

    Priority (highest → lowest):
        explicit CLI flag  >  config file value  >  argparse default
    """
    # ── inject --config without breaking the original parser ─────────────────
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument('--config', type=str, default=None,
                             help='Path to YAML config file')
    cfg_args, remaining = cfg_parser.parse_known_args()

    if cfg_args.config is not None:
        config_path = Path(cfg_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        # YAML values become argparse defaults (CLI can still override them)
        parser.set_defaults(**cfg)

    # ── add --config to the main parser so it appears in --help and args ─────
    parser.add_argument('--config', type=str, default=None,
                         help='Path to YAML config file (values overridden by CLI flags)')

    return parser.parse_args()
