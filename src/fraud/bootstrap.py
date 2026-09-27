"""Fetch the public ULB credit-card dataset when no local raw CSV exists."""

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip"
DATA_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def ensure_raw_data(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        archive = Path(temporary) / "creditcard.zip"
        urllib.request.urlretrieve(DATA_URL, archive)
        extracted = Path(temporary) / "creditcard.csv"
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open("creditcard.csv") as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        with extracted.open("rb") as data:
            digest = hashlib.file_digest(data, "sha256").hexdigest()
        if digest != DATA_SHA256:
            raise ValueError(f"Downloaded dataset SHA-256 mismatch: {digest}")
        extracted.replace(path)
    return path
