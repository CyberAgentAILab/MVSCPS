from collections import defaultdict
from typing import Any, Dict


class Registry:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def register(self, kind: str, name: str = None):
        def deco(obj):
            key = name or obj.__name__
            if key in self._store[kind]:
                raise KeyError(f"[{kind}] '{key}' already registered")
            self._store[kind][key] = obj
            return obj
        return deco

    def get(self, kind: str, name: str):
        return self._store[kind][name]

    def build(self, kind: str,  cfg: dict, name=None):
        if name is None:
            cls = self.get(kind, cfg["name"])
        else:
            cls = self.get(kind, name)
        return cls(cfg)

REG = Registry()