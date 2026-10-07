"""Config loading, seeding and CSV logging."""
import copy
import csv
import os
import random

import numpy as np
import yaml


def load_config(path, overrides=None):
    """Read a YAML config and apply "a.b.c=value" overrides (values parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"unknown config section {p!r} in {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_copy(cfg):
    return copy.deepcopy(cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device(name="auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class CSVLogger:
    """Write rows (dicts) to a CSV file; the header comes from the first row.

    The file stays open for the whole run. Re-opening it for every row fails on folders
    synced by OneDrive or Dropbox, which lock a file for a moment after each change.
    """

    def __init__(self, path):
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = None

    def log(self, row):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()), extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()
