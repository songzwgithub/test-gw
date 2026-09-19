from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    root: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' must be a mapping")
        return value

    def resolve(self, value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def outputs(self) -> Path:
        return self.resolve(self.raw.get("outputs", "outputs"))


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project = raw.get("project", {})
    root_value = project.get("root", ".")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return ProjectConfig(path=path, root=root, raw=raw)
