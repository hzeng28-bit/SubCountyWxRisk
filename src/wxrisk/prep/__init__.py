"""Data preprocessing as its own, reproducible pipeline stage: raw -> clean -> normalized.

    raw         downloaded files, untouched (fingerprinted by data.lock.json)
    clean       typed, key-normalised, null/sentinel/range rules applied, deterministic row order  (prep.clean, prep.build)
    normalized  scaling parameters fitted on the training period only, applied as a pure function  (prep.normalize)

Every output has a `.prep.json` sidecar (input content hash, spec hash, row accounting, output content hash), and
`build.verify` re-derives the output from the raw file and compares. See docs/DATA_PREP.md.
"""

from .spec import Col, TableSpec  # noqa: F401
