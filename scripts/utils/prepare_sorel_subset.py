#!/usr/bin/env python3
# scripts/utils/prepare_sorel_subset.py
#
# Build a small (X,y) NPZ + metadata CSV subset from SOREL-20M meta.db manifest
# by extracting EMBER-style features from the SOREL ember_features LMDB.
#
# LMDB values are zlib-compressed msgpack; features live under key 0.

import argparse
from pathlib import Path
import binascii
import zlib

import lmdb
import msgpack
import numpy as np
import pandas as pd


ZLIB_HEADERS = (b"\x78\x9c", b"\x78\xda", b"\x78\x01")


def maybe_decompress(buf: bytes) -> bytes | None:
    if buf is None:
        return None
    if len(buf) >= 2 and buf[:2] in ZLIB_HEADERS:
        try:
            return zlib.decompress(buf)
        except Exception:
            return None
    return buf


def unpack_feature_vector(raw: bytes) -> np.ndarray | None:
    """
    Decode a SOREL ember_features LMDB value into a 1D float array.
    Observed format: zlib-compressed msgpack dict with key 0 -> vector-like.
    """
    raw = maybe_decompress(raw)
    if raw is None:
        return None

    try:
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        return None

    if not isinstance(obj, dict) or 0 not in obj:
        return None

    x = obj[0]

    # common cases
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float32)
    elif isinstance(x, (bytes, bytearray, memoryview)):
        # if packed as binary float array; interpret as float32
        try:
            arr = np.frombuffer(bytes(x), dtype=np.float32)
        except Exception:
            return None
    else:
        try:
            arr = np.asarray(x, dtype=np.float32)
        except Exception:
            return None

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if not np.all(np.isfinite(arr)):
        # if it contains NaNs/Infs, still allow but replace with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return arr


def infer_dim(manifest: pd.DataFrame, lmdb_dir: Path, scan: int, dbname: bytes | None = None) -> int | None:
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=256)
    db = env.open_db(dbname) if dbname else None

    dims = []
    with env.begin(write=False, db=db) as txn:
        for sha in manifest["sha256"].astype(str).head(scan):
            v = txn.get(sha.encode("ascii"))
            if v is None:
                continue
            arr = unpack_feature_vector(v)
            if arr is None:
                continue
            d = int(arr.shape[0])
            dims.append(d)
            if len(dims) >= 2000:
                break

    env.close()
    if not dims:
        return None

    # pick the most common dimension
    vals, counts = np.unique(np.array(dims), return_counts=True)
    return int(vals[np.argmax(counts)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV from make_sorel_manifest.py (rowid,sha256,is_malware,...)")
    ap.add_argument("--lmdb", required=True, help="path to SOREL ember_features directory containing data.mdb")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-meta-csv", required=True)
    ap.add_argument("--limit", type=int, default=2000, help="max rows to extract into NPZ")
    ap.add_argument("--scan", type=int, default=50000, help="rows to scan for inferring target dim")
    ap.add_argument("--expect-dim", type=int, default=None, help="force feature dimension")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--npz-key", type=str, default=None, help="optional LMDB named db (rare); usually None")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    manifest = pd.read_csv(args.manifest)
    if "sha256" not in manifest.columns or "is_malware" not in manifest.columns:
        raise ValueError("manifest must contain sha256 and is_malware columns")

    lmdb_dir = Path(args.lmdb)
    if lmdb_dir.is_dir() and not (lmdb_dir / "data.mdb").exists():
        raise FileNotFoundError(f"Expected {lmdb_dir}/data.mdb")

    dbname = args.npz_key.encode("utf-8") if args.npz_key else None

    target_dim = args.expect_dim
    if target_dim is None:
        target_dim = infer_dim(manifest, lmdb_dir, args.scan, dbname=dbname)
        if target_dim is None:
            raise RuntimeError("Could not infer feature dimension; no decodable rows found")
        print(f"[info] target_dim={target_dim} (use --expect-dim to override)")

    # shuffle manifest rows so we don’t get stuck on a bad block
    idxs = np.arange(len(manifest))
    rng.shuffle(idxs)

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=256)
    db = env.open_db(dbname) if dbname else None

    X_rows = []
    y_rows = []
    meta_rows = []

    missing = 0
    bad_decode = 0
    dim_mismatch = 0

    with env.begin(write=False, db=db) as txn:
        for i in idxs:
            if len(X_rows) >= args.limit:
                break

            row = manifest.iloc[int(i)]
            sha = str(row["sha256"])
            y = int(row["is_malware"])

            v = txn.get(sha.encode("ascii"))
            if v is None:
                missing += 1
                continue

            arr = unpack_feature_vector(v)
            if arr is None:
                bad_decode += 1
                continue

            if int(arr.shape[0]) != int(target_dim):
                dim_mismatch += 1
                continue

            X_rows.append(arr.astype(np.float32, copy=False))
            y_rows.append(y)
            meta_rows.append(row.to_dict())

    env.close()

    if not X_rows:
        print(f"No rows extracted from LMDB (missing={missing} bad_decode={bad_decode} dim_mismatch={dim_mismatch}).")
        return

    X = np.vstack([x.reshape(1, -1) for x in X_rows]).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int32)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y)

    out_meta = Path(args.out_meta_csv)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta_rows).to_csv(out_meta, index=False)

    print(f"[ok] extracted {len(X_rows)} rows  d={X.shape[1]}")
    print(f"[ok] wrote {out_npz}")
    print(f"[ok] wrote {out_meta}")
    print(f"[stats] missing={missing} bad_decode={bad_decode} dim_mismatch={dim_mismatch}")


if __name__ == "__main__":
    main()
