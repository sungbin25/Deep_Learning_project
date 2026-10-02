"""Download original authors' CC BY 4.0 archive, preserving its README/LICENSE."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
URL = "https://codeload.github.com/sea3551/palm-sEMG-doorknob-filtered/zip/refs/heads/main"

def main():
    data = ROOT / "data"
    if list(data.rglob("*.csv")):
        print("Existing CSV files found; keeping them unchanged.")
        return
    blob = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts[1:]
            if not parts or name.endswith("/"):
                continue
            target = data.joinpath(*parts).resolve()
            if not target.is_relative_to(data.resolve()):
                raise ValueError(f"Unsafe archive path: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    (data / "download_metadata.json").write_text(json.dumps({"url":URL,"downloaded_utc":datetime.now(timezone.utc).isoformat(),"archive_sha256":hashlib.sha256(blob).hexdigest()},indent=2),encoding="utf-8")
    print(f"Downloaded {len(list(data.rglob('*.csv')))} CSV files.")

if __name__ == "__main__":
    main()
