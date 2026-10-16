"""Refuse distributions that are incomplete or that ship what they must not.

    python scripts/check_dist.py dist 0.1.0

Run by `.github/workflows/release.yml` after the build and by
`scripts/ci-local.sh --full`. The folder must hold exactly the wheel and the
sdist for the version. The wheel must carry the settings file, `py.typed`, the
built configurator UI (gitignored, so a build without `npm run build` has no
UI) and the LICENSE and NOTICE files; neither file may carry tests, a
`conftest.py`, `node_modules`, a `.env` other than `.env.example`, or the
Kubernetes overlay that holds an operator's secrets (`deploy/k8s/overlays/local/`).
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

REQUIRED = [
    "lasagna/settings.lasagna.toml",
    "lasagna/py.typed",
    "lasagna/configurator/frontend/dist/index.html",
]
UNWANTED = re.compile(
    r"(^|/)(tests/|conftest\.py|node_modules/|\.env$|\.env\.(?!example)|deploy/k8s/overlays/local/)"
)


def problems(dist: Path, version: str) -> list[str]:
    wheel = dist / f"lasagna_rag-{version}-py3-none-any.whl"
    sdist = dist / f"lasagna_rag-{version}.tar.gz"
    found = sorted(p.name for p in dist.iterdir())
    if found != sorted([wheel.name, sdist.name]):
        return [f"{dist} holds {found}, expected only {wheel.name} and {sdist.name}"]
    names = zipfile.ZipFile(wheel).namelist()
    missing = [n for n in REQUIRED if n not in names]
    missing += [
        f"*.dist-info/licenses/{f}"
        for f in ("LICENSE", "NOTICE")
        if not any(n.endswith(f".dist-info/licenses/{f}") for n in names)
    ]
    with tarfile.open(sdist) as archive:
        shipped = [n for n in names + archive.getnames() if UNWANTED.search(n)]
    print(f"{wheel.name}: {len(names)} files")
    out = []
    if missing:
        out.append(f"missing {missing}")
    if shipped:
        out.append(f"should not ship {shipped[:20]}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    found = problems(Path(argv[0]), argv[1])
    for line in found:
        print(f"::error::{line}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
