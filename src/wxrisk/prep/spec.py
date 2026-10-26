"""Declarative description of how one raw table becomes its clean table.

The spec is data, not code: it is hashed into the sidecar, so changing any rule changes the output's identity.

Missing values come in three kinds and each has an explicit rule (the null-handling policy, docs/DATA_PREP.md):
  structural  a null that means something (no termination date = still running)         nulls="flag"  (keep null, add flag)
  unknown     the source says "unknown" with a code (foundationType '0'), or the value is out of range
              -> `sentinel` / `lo`,`hi` turn it into null, nulls="flag" records that it was there
  unusable    a key that cannot be used to join (null / malformed block-group code)       nulls="drop"  (row removed, counted)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

KINDS = ("key", "str", "int", "long", "float", "bool", "date")
NULLS = ("keep", "flag", "drop")


@dataclass(frozen=True)
class Col:
    src: str
    kind: str
    name: str | None = None  # output name, default = src
    nulls: str = "keep"
    sentinel: tuple = ()  # source values that mean "unknown" (compared after typing/strip/upper)
    lo: float | str | None = None  # values outside [lo, hi] become null (counted); dates as ISO strings
    hi: float | str | None = None
    width: int | None = None  # key: zero-pad to this many digits; anything else is malformed
    upper: bool = False  # str: upper-case
    parse: str | None = None  # date stored as text: strptime format, applied to the first len(format-rendered) chars (ISO prefix)

    def __post_init__(self):
        assert self.kind in KINDS and self.nulls in NULLS, self
        assert self.kind != "key" or self.width, "a key needs a width"

    @property
    def out(self) -> str:
        return self.name or self.src


@dataclass(frozen=True)
class TableSpec:
    name: str
    source: str  # path or glob relative to $WXRISK_DATA (a glob = one table in several files, e.g. per year)
    cols: tuple[Col, ...]
    version: int = 1  # bump when the cleaning code changes meaning
    expect_unique: tuple[str, ...] = ()  # output columns that must jointly be unique (empty = no check)
    derived: tuple[str, ...] = ()  # derived columns, implemented in clean.DERIVED (named so the spec stays hashable data)
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    def digest(self) -> str:
        blob = json.dumps({"v": self.version, "name": self.name, "source": self.source,
                           "cols": [asdict(c) for c in self.cols], "derived": list(self.derived)}, sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()
