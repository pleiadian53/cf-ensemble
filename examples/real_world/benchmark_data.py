"""
Dataset loaders for the realistic CF-Ensemble benchmark.

Each loader returns ``(X, y)``: a float32 feature matrix and binary labels in {0, 1}.
Downloads are cached under ``data_root`` (on the pod: the network volume, so they
persist across pods).

Registered datasets:
- ``protein_homo``: KDD Cup 2004 protein homology (145,751 × 74, ~0.9% positives),
  via imbalanced-learn's Zenodo mirror. Pipeline smoke test.
- ``coffin_splice``: human acceptor splice sites from Sonnenburg & Franc (2010),
  "COFFIN" (http://sonnenburgs.de/soeren/item/coffin/): 141-nt windows, labels ±1.
  We use the 1M-example training subset (the 10M/50M sets are skipped on extract).
"""

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

LOADERS: Dict[str, Callable[[Path], Tuple[np.ndarray, np.ndarray]]] = {}


def register(name: str):
    def _wrap(fn):
        LOADERS[name] = fn
        return fn
    return _wrap


@register("protein_homo")
def load_protein_homo(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    from imblearn.datasets import fetch_datasets

    cache = data_root / "imblearn"
    cache.mkdir(parents=True, exist_ok=True)
    ds = fetch_datasets(data_home=str(cache), filter_data=("protein_homo",))["protein_homo"]
    y = (ds.target == 1).astype(np.int8)  # imblearn encodes classes as -1 / +1
    return ds.data.astype(np.float32), y


COFFIN_URL = "http://sonnenburgs.de/soeren/projects/coffin/splice_data.tar.xz"
COFFIN_MD5 = "4bd0be48ae163ea9f4893e5a5c423c79"  # compressed archive, from the COFFIN page
COFFIN_STEM = "H_sapiens_acc_all_examples_plain"


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_coffin(root: Path, max_member_mb: int = 500) -> Path:
    """Download + verify the COFFIN archive; extract members up to ``max_member_mb``.

    Streams the tar so the 10M/50M-example files (GBs each) are never written out.
    Idempotent: a ``.extracted`` marker skips work on later runs.
    """
    root.mkdir(parents=True, exist_ok=True)
    archive, marker = root / "splice_data.tar.xz", root / ".extracted"
    if marker.exists():
        return root / "data"
    if not archive.exists() or _md5(archive) != COFFIN_MD5:
        print(f"[coffin] downloading {COFFIN_URL} (1.7 GB)")
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(COFFIN_URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 24)
        tmp.rename(archive)
        if _md5(archive) != COFFIN_MD5:
            raise IOError(f"md5 mismatch for {archive}")
    print("[coffin] extracting members <= %d MB" % max_member_mb)
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            keep = member.isfile() and member.size <= max_member_mb * 1e6
            print(f"[coffin]   {'keep' if keep else 'skip'} {member.name} ({member.size / 1e6:,.0f} MB)")
            if keep:
                tar.extract(member, path=root, filter="data")
    marker.touch()
    return root / "data"


def encode_dna(seqs: np.ndarray, k: int = 3) -> np.ndarray:
    """One-hot of every position plus k-mer counts of each half of the window.

    ``seqs`` is a (n, L) uint8 array of ASCII bases. Returns float32 (n, 4L + 2·4^k).
    Positional one-hot suits linear/tree models; k-mer spectra give position-free
    composition features, so base models see complementary views.
    """
    lut = np.full(256, 4, dtype=np.int64)  # anything not ACGT (e.g. N) -> 4
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
        lut[b + 32] = i  # lowercase
    idx = lut[seqs]  # (n, L)
    n, L = idx.shape
    onehot = np.zeros((n, L, 4), dtype=np.float32)
    valid = idx < 4
    onehot[np.nonzero(valid) + (idx[valid],)] = 1.0

    codes = np.zeros((n, L - k + 1), dtype=np.int64)
    ok = np.ones((n, L - k + 1), dtype=bool)
    for j in range(k):
        part = idx[:, j:L - k + 1 + j]
        codes = codes * 4 + np.minimum(part, 3)
        ok &= part < 4
    half = (L - k + 1) // 2
    spectra = []
    for sl in (slice(0, half), slice(half, None)):
        c, m = codes[:, sl], ok[:, sl]
        flat = (np.arange(n)[:, None] * 4 ** k + c)[m]  # (row, kmer) -> one bin index
        counts = np.bincount(flat, minlength=n * 4 ** k).reshape(n, 4 ** k)
        spectra.append(counts.astype(np.float32))
    return np.hstack([onehot.reshape(n, -1)] + spectra)


@register("coffin_splice")
def load_coffin_splice(data_root: Path, size_tag: str = "1000000") -> Tuple[np.ndarray, np.ndarray]:
    data_dir = fetch_coffin(data_root / "coffin")
    seq_files = sorted(p for p in data_dir.glob(f"{COFFIN_STEM}_139-279*.fasta")
                       if size_tag in p.name or (size_tag == "1000000" and "1e6" in p.name))
    if not seq_files:
        raise FileNotFoundError(f"no {size_tag}-example sequence file in {data_dir}: "
                                f"{sorted(p.name for p in data_dir.iterdir())}")
    seq_path = seq_files[0]
    lines = seq_path.read_bytes().split()
    seqs = np.frombuffer(b"".join(lines), dtype=np.uint8).reshape(len(lines), -1)
    # The label file is the one whose line count matches the sequences; try size-tagged names first
    candidates = sorted(data_dir.glob(f"{COFFIN_STEM}_*.label"), key=lambda p: size_tag not in p.name)
    for lab_path in candidates:
        raw = lab_path.read_bytes()
        if raw.count(b"\n") + (not raw.endswith(b"\n")) == len(seqs):
            labels = np.array(raw.split(), dtype=np.float32)
            break
    else:
        raise FileNotFoundError(f"no label file with {len(seqs):,} lines in {data_dir}")
    print(f"[coffin] {seq_path.name} + {lab_path.name}: {len(seqs):,} × {seqs.shape[1]} nt")
    return encode_dna(seqs), (labels > 0).astype(np.int8)


def load_dataset(
    name: str,
    data_root: str = "data",
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a registered dataset, optionally stratified-subsampled to ``max_rows``."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choices: {', '.join(LOADERS)}")
    X, y = LOADERS[name](Path(data_root))
    if max_rows is not None and max_rows < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X, y
