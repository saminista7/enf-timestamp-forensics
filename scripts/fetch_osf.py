"""Discover and download power-grid-frequency EDR recordings from OSF.

Two phases, deliberately separate:

  discover  walk the OSF file tree and write data/manifest/osf_catalogue.yaml
            (name, file id, bytes, download url, sha256-as-published)
  fetch     download what the catalogue lists, verify sizes and hashes,
            and write data/manifest/data_manifest.yaml

Keeping them separate means the catalogue is a committed, reviewable record of
what existed when we looked, and a later re-fetch can prove nothing moved.

The OSF v2 API quirk worth knowing: folder contents are listed via
    /v2/nodes/<node>/files/osfstorage/<folder_id>/
and NOT via /v2/files/<folder_id>/files/, which 404s.

Usage
-----
    python scripts/fetch_osf.py discover --node by5hu
    python scripts/fetch_osf.py fetch --only ZA02,KR01,US_CO01
    python scripts/fetch_osf.py fetch --all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifest"
RAW_DIR = REPO / "data" / "raw"

API = "https://api.osf.io/v2"
DOWNLOAD = "https://osf.io/download/{file_id}/"
TIMEOUT = 120
RETRIES = 4
BACKOFF = 3.0

# The folder holding one subfolder per measurement site.
FREQUENCY_DATA = "FrequencyData"


@dataclass
class OsfFile:
    site: str
    name: str
    file_id: str
    size_bytes: int | None
    published_md5: str | None
    published_sha256: str | None
    download_url: str
    node: str = "by5hu"
    rel_dir: str = FREQUENCY_DATA  # directory under data/raw/osf_<node>/, as published

    @property
    def local_path(self) -> Path:
        # rel_dir is the published folder path, so the local tree mirrors OSF exactly
        return RAW_DIR / f"osf_{self.node}" / Path(self.rel_dir) / self.name


def _get(url: str, params: dict | None = None) -> dict:
    """GET with retries. OSF rate-limits and occasionally 502s; a failed walk
    that silently returns a short list would corrupt the catalogue, so a
    persistent failure raises rather than degrades."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(BACKOFF * (attempt + 2))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last = exc
            time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last


def _list_folder(node: str, folder_id: str | None) -> list[dict]:
    """Return every entry in an osfstorage folder, following pagination.

    NOTE the endpoint shape. Folder contents live at
        /v2/nodes/<node>/files/osfstorage/<folder_id>/
    The seemingly natural /v2/files/<folder_id>/files/ returns
        {"errors":[{"detail":"Not found."}]}
    which reads as "empty folder" rather than "wrong route". That 404 is how you end up
    with a quietly incomplete catalogue: it cost me 12 of the 32 sites the first time.
    """
    url = f"{API}/nodes/{node}/files/osfstorage/"
    if folder_id:
        url += f"{folder_id}/"
    out: list[dict] = []
    params: dict | None = {"page[size]": 100}
    while url:
        payload = _get(url, params)
        out.extend(payload.get("data", []))
        url = payload.get("links", {}).get("next")
        params = None  # the `next` link already carries them
    return out


def _as_file(entry: dict, site: str, rel_dir: str, node: str) -> OsfFile:
    attrs = entry["attributes"]
    hashes = attrs.get("extra", {}).get("hashes", {}) or {}
    return OsfFile(
        site=site,
        name=attrs["name"],
        file_id=entry["id"],
        size_bytes=attrs.get("size"),
        published_md5=hashes.get("md5"),
        published_sha256=hashes.get("sha256"),
        download_url=DOWNLOAD.format(file_id=entry["id"]),
        node=node,
        rel_dir=rel_dir,
    )


def _walk(
    node: str, folder_id: str | None, prefix: str, depth: int, max_depth: int
) -> list[OsfFile]:
    """Depth-first walk collecting every file, with `site` set to its folder path.

    Used for nodes whose tree is not one-folder-per-site (m43tg nests
    Data/<synchronous area>/<source>/<year>/...).
    """
    files: list[OsfFile] = []
    for entry in _list_folder(node, folder_id):
        attrs = entry["attributes"]
        name = attrs["name"]
        path = f"{prefix}/{name}" if prefix else name
        if attrs["kind"] == "file":
            files.append(_as_file(entry, prefix or ".", prefix or ".", node))
        elif depth < max_depth:
            print(f"  {'  ' * depth}{name}/", file=sys.stderr)
            files.extend(_walk(node, entry["id"], path, depth + 1, max_depth))
    return files


def discover(node: str, *, recursive: bool = False, max_depth: int = 6) -> list[OsfFile]:
    """Catalogue a node.

    by5hu has one folder per measurement site under FrequencyData, so the default
    path handles it directly. m43tg (the TSO mirror) nests several levels deep and
    needs --recursive.
    """
    if recursive:
        print(f"recursive walk of node {node}", file=sys.stderr)
        files = _walk(node, None, "", 0, max_depth)
        sites = {f.site for f in files}
        print(f"{len(files)} files across {len(sites)} folders", file=sys.stderr)
        return files

    root = _list_folder(node, None)
    freq = next((e for e in root if e["attributes"]["name"] == FREQUENCY_DATA), None)
    if freq is None:
        raise RuntimeError(
            f"{FREQUENCY_DATA} folder not found in node {node}; "
            f"saw {[e['attributes']['name'] for e in root]}. Try --recursive."
        )

    sites = sorted(
        (e["attributes"]["name"], e["id"])
        for e in _list_folder(node, freq["id"])
        if e["attributes"]["kind"] == "folder"
    )
    print(f"{len(sites)} site folders in {node}/{FREQUENCY_DATA}", file=sys.stderr)

    files: list[OsfFile] = []
    for site, site_id in sites:
        for entry in _list_folder(node, site_id):
            if entry["attributes"]["kind"] != "file":
                continue
            files.append(_as_file(entry, site, f"{FREQUENCY_DATA}/{site}", node))
        print(f"  {site:28s} {len(files):4d} files so far", file=sys.stderr)
    return files


def catalogue_path(node: str) -> Path:
    # per-node, so cataloguing m43tg cannot silently overwrite by5hu
    legacy = MANIFEST_DIR / "osf_catalogue.yaml"
    if node == "by5hu" and legacy.exists():
        return legacy
    return MANIFEST_DIR / f"osf_catalogue_{node}.yaml"


def write_catalogue(files: list[OsfFile], node: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = catalogue_path(node)
    doc = {
        "node": node,
        "api": API,
        "discovered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sites": len({f.site for f in files}),
        "n_files": len(files),
        "files": [asdict(f) for f in files],
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"wrote {out} ({doc['n_sites']} sites, {doc['n_files']} files)", file=sys.stderr)
    return out


def load_catalogue(node: str) -> tuple[str, list[OsfFile]]:
    path = catalogue_path(node)
    if not path.exists():
        raise SystemExit(f"no catalogue at {path}; run `discover --node {node}` first")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = []
    for f in doc["files"]:
        f.setdefault("node", doc.get("node", node))
        f.setdefault("rel_dir", f"{FREQUENCY_DATA}/{f['site']}")
        files.append(OsfFile(**f))
    return doc["node"], files


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_one(f: OsfFile, *, overwrite: bool = False) -> dict:
    dest = f.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        if f.size_bytes is None or dest.stat().st_size == f.size_bytes:
            return {"status": "cached", "path": str(dest), "bytes": dest.stat().st_size}
        print(f"  size mismatch, re-downloading {f.site}/{f.name}", file=sys.stderr)

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(f.download_url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for block in r.iter_content(chunk_size=1 << 20):
                fh.write(block)
    tmp.replace(dest)

    got = dest.stat().st_size
    if f.size_bytes is not None and got != f.size_bytes:
        raise RuntimeError(f"{f.site}/{f.name}: expected {f.size_bytes} bytes, got {got}")
    return {"status": "downloaded", "path": str(dest), "bytes": got}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="walk OSF and write the catalogue")
    d.add_argument("--node", default="by5hu")
    d.add_argument(
        "--recursive",
        action="store_true",
        help="depth-first walk; needed for m43tg, whose tree nests by area/source/year",
    )
    d.add_argument("--max-depth", type=int, default=6)

    g = sub.add_parser("fetch", help="download catalogued files")
    g.add_argument("--node", default="by5hu")
    g.add_argument("--only", default=None, help="comma-separated site codes or path substrings")
    g.add_argument("--all", action="store_true")
    g.add_argument("--ext", default=None, help="only files with this extension, e.g. .csv")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--hash", action="store_true", help="compute sha256 of each file (slow)")

    args = ap.parse_args()

    if args.cmd == "discover":
        files = discover(args.node, recursive=args.recursive, max_depth=args.max_depth)
        write_catalogue(files, args.node)
        return

    node, files = load_catalogue(args.node)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        files = [f for f in files if f.site in wanted or any(w in f.site for w in wanted)]
    elif not args.all:
        raise SystemExit("pass --only SITE,SITE or --all")
    if args.ext:
        files = [f for f in files if f.name.lower().endswith(args.ext.lower())]
    if not files:
        raise SystemExit("no files matched the filters")

    records = []
    total = 0
    for f in files:
        res = fetch_one(f, overwrite=args.overwrite)
        total += res["bytes"]
        rec = {
            "node": f.node,
            "site": f.site,
            "name": f.name,
            "rel_dir": f.rel_dir,
            "file_id": f.file_id,
            "download_url": f.download_url,
            "bytes": res["bytes"],
            "status": res["status"],
            "published_md5": f.published_md5,
        }
        if args.hash:
            rec["sha256"] = sha256_of(Path(res["path"]))
        records.append(rec)
        print(
            f"{res['status']:11s} {f.site}/{f.name}  {res['bytes'] / 1e6:.1f} MB", file=sys.stderr
        )

    out = MANIFEST_DIR / "data_manifest.yaml"
    existing = {}
    if out.exists():
        doc = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        existing = {(r.get("node", "by5hu"), r["site"], r["name"]): r for r in doc.get("files", [])}
    for r in records:
        existing[(r["node"], r["site"], r["name"])] = r

    out.write_text(
        yaml.safe_dump(
            {
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "nodes": sorted({r.get("node", "by5hu") for r in existing.values()}),
                "files": [existing[k] for k in sorted(existing)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(records)} files, {total / 1e9:.2f} GB; manifest -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
