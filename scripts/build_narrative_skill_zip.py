# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild opencontext/server/resources/narrative-to-base-events.zip
from .claude/skills/narrative-to-base-events/.

Run this whenever the skill source changes, then commit both the source
files and the regenerated zip.

The zip is byte-deterministic: entries are written in sorted order with a
fixed timestamp and fixed permissions, so re-running this script with
unchanged source content produces byte-identical output (clean no-op diff;
enables a CI drift check).
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".claude" / "skills" / "narrative-to-base-events"
DST = ROOT / "opencontext" / "server" / "resources" / "narrative-to-base-events.zip"

# Fixed epoch so the archive bytes depend only on file contents, not on
# filesystem mtimes (which vary across clones, editor saves, etc.).
_FIXED_DATE_TIME = (2024, 1, 1, 0, 0, 0)
_FIXED_MODE = 0o644 << 16  # rw-r--r-- as external_attr on unix


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Source skill directory not found: {SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DST, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if not path.is_file():
                continue
            # Skip Python bytecode caches — they pollute the bundle and vary
            # between runs (different interpreter versions).
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            arcname = Path("narrative-to-base-events") / path.relative_to(SRC)
            info = zipfile.ZipInfo(arcname.as_posix(), date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _FIXED_MODE
            zf.writestr(info, path.read_bytes())

    print(f"Wrote {DST.relative_to(ROOT)} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
