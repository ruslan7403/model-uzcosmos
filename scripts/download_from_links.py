#!/usr/bin/env python3
"""Download and extract dataset from a list of URLs (zip and txt).

No API token required. Extracts zip files into the output directory
so that class subdirectories are preserved (data/mapillary/class_name/*.jpg).

Usage:
    python scripts/download_from_links.py --output-dir data/mapillary
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Dataset download URLs (no Mapillary token needed)
DATASET_URLS = [
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8RSqTVAOhQUZHE5NgIL-tgliMqFe4h3ieVXSDNI-ilkh922n_TtB2VpiW-SzAHEYLiqk3Y3FU8pM-DskkOMCvFGh9RJAxxqi3RI2JXDvbQfq4xYwVstPYKlT44GhxzbDbdW_OG0lyY.txt?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adkf0QW3jQuJg9vRQudIFxERexeIlNJWnlY6LnEYKDQv7JHNJhZ_D8lsJqaNZh3O-B8&ccb=10-5&oh=00_AfsXaLhmgowJY69KQlQfCH79bMCXM1YceO__UB87MvHG8A&oe=69CA1D45&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An90x89nHvauCK1fqMJ8110KeTjNo5Si7rzhvwIMCu5xI9_GhWBGOIXaFvu6o53NuNpBMzdC9qsjAVR8sLv8m6WoFfn6Qd4NjMYKNW4NCKVp6gx3MhZtwf3cZR94wFhou5lPI0hGUw.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdmnIHbGPbvbfA4T6e2RZ2ljiljqeDAi_NuMctzziXbGNEs6q4V26tiXUdYsI9zl71I&ccb=10-5&oh=00_Afsex7q0w02AScH0fhLzl54ZKNoQ63wlPLN4cSODsLSbCA&oe=69CA0E88&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9eB8zXYW473CiQYW9CPvGx1Ho-fkQvkini3ddExpFOz47aWs4ydBSvK-ZhOPu7ikASQmZvX0zyXhmJzBr6CDZE5ZkUhvJ44h7mV2NT4cSRbR837J9mHJosreQRJJdGaVDR26EAjLPL.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdnlDs4vn35vQu5MVm8KE620i9f23HfcgWxYXMFS658b1F6xv4c3wL2QvSrrro_v9bY&ccb=10-5&oh=00_AftAgfJ0pycNuJxTWr-y_bLjgXBecrkoOhb04pVhxAc2Jw&oe=69CA181C&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_WKGcw-ICowA_xAgTEU_E-pAYdybyzT-9Pwi8JtelanWnNRKONV1DTAZPEAsGNDWlFpYDi16km1stDN47ip-quE77cfkv3aERdMIRahysGgspb6DlCgrabPSTFJI3tZ9EMRatRC6ZmjQytVcY.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlhNFHfVcTLlCmkFo1lcDRyjjEmPSutbfma5a5lfxnO-Yx065sy5mj7NVYVx1-cw6Y&ccb=10-5&oh=00_Afv_rbOWR_itZKG9cys_6waZ5FNQ0ZMifIPo99L_Kau_8Q&oe=69CA1472&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8VtaI-ldaSOc5HcLFVQ6SPvHDt50hLG1kga0nUfswldLu1J9dsOx6ynZicRUuXR_TvsczpplOqQEa7ppT4JwUzI0ZNQCHmhtkfT5tjdNJY55Ud6eXplvq59PjOx55d2EbIxYpO9vhR-BcflQ.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdkzDWnefJ-Db_muEXHMUM4TzFwP64wozlaor3oJLZLMXZT4BMd4ICL3TtkdsCNTgkU&ccb=10-5&oh=00_AfvfoxCPHSLe4U0PWSD-iCb97aXxFpg-1-YkEC9QTTHwLA&oe=69CA265E&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9BwhO8zTAy6jGrwy71BjIBtMK8K5RkpIJguP7DVnpJK2TfKDlfxXj8mCxRJss4zzfaaKi2idqbQOtYJ740TPCI7w7hL8V7goknzuO0ZFPLywDCKIB7i64lCiSUNYXLqeS8mC7EkiU5hYAfOcI.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlrNTk2ctb2KXEUiaw9oVBaVZwuLkqQgRrPx7YVXtKVa8NoVWryXiNj2c16KtT6KT8&ccb=10-6&oh=00_AfurDewpgTt5Biy45uuR9oxb7hmsJMWX6O1OqOJ0pjkLjA&oe=69CA0FDB&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9tl3SwDwRFQ2z9tAag26brYamdQZmHnVoxfNTz_Iass-zZLWM-HryqW44UeqbLWd-EkXVIP-ZQQfg3F7dmQYlnu1wjzCARviaJMBHgtLH4gTAeW6msFbEXA3_NIZBtdP7Gg8dt5Ewl.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlW-xoWKeve6GWxjBw22h9Yddkjd5Fsnro6dNyPiS7GyllE8wIgeJkD1Py4dIPTxDY&ccb=10-5&oh=00_AfvabCHO64OhX5tU27xS6C2aczj2HRiFbphpgHkOyEDJzA&oe=69CA2F67&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9FlyymG8PIT47PpdF-es5LyB3G1VvhyKar744ioBXCcFVQDaTVBq1LTvq1vz61u49tefGZb03n-mQ_PWKPIX8ZrMfRI09WMTygDexYMZJ6VkTWJK_7FTY47dC9KeKQG8RG3oGU6Dhw.txt?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlmzuPRWIcDay_D4Kgg24mq_6HQLGB57pQzAtggBbvt_FtqogZlXP7JzCCBuX-zrt0&ccb=10-5&oh=00_AfuYJ1FS0mDr7oNLmLdoYlEg-kWzxdJ1ySlZmYXa9j85uA&oe=69CA2EC3&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8dmcHh74W4QOmBjSUnIgjyIDl7pVOj8ym708qotPkG4w_03qYL9KUrHpljvhZgYfNJp82zqzc5ZNAawW6W-huIX9hXtE0Q1wEbbbJWyP2Qgfosi_118TvAPBpKxis6XM8RroqJwg.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adlbe1ae0reUeGDmAFyIFd1zitVuvjjenMio3n-XdXD09HPfKKY4GT6iBZUjBJR6XnU&ccb=10-5&oh=00_AfvOkp1p6sF1VNkdakpCVGwIu_BtFNzcLiA6_RA5gEKSxw&oe=69CA3550&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_XCkEh3XjG7bV8c4h5QX-Qodh85URfwXB83u_txLJaEcDmFH6VOUG0O2T8U5WZt7mpEa-EczZiupLH93KsSXfDAIPq8HjPJn1q_oRSBA3ufZus7MnJ_CGs1O166IJiczbnq9J-0NOZ_G5mR8c.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adn0F6YkwqRkAKZXG-F9MXlclEBmpnrBAxFey4wz4WWOtftacrdHmDk455my2XdnLd8&ccb=10-5&oh=00_Afs6T6FcCK0NRloM4pD9wozCJWvS7ScJI6G_cEMb6pZKyg&oe=69CA18C0&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8bwxH70h7HKMm7H2IGV4ptYgE55ozIcf4_wWboduU4ToWxKTnWIelPbSqQ3C1RxKoRTjsOfB9_gzdrv6Oq6qLxEJ1IfOuOXvv6btXEwLfAw2feJOla96rwgmmQyhf-AzgqpS1yIKeydT8zkA.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adm5rfWythwb17S-ZPg_IgaLdFoxdIZCNTx0EKSZ4Zc8Gzvq6Q5p65qR7rGed1cKynE&ccb=10-5&oh=00_AftvVpRy1PYyygjqmo9Rc7SThX6axV_eZkBLYy41TdSeDw&oe=69CA228B&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An-nQsgt_uWGakflQM3LbwL32CaQBwAqrR1r2jdVCHNoR34x1v6LHlvlpTQ9CM86r3tREP05tP049I58J7utUv4vefGA3XD2Up-fJ5cXubNGDCglkw-haZwHnvQR-QtqOisk_5IOaEUK62MYBAE.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdmsNfWK4XeuoxPyZAWOJ_kgxOMIEIUKSnHDxz7CHFgKNJBqZQC2OJ6cVuDlsou_Lj4&ccb=10-5&oh=00_AfshK6BNGusfB4D9ImWuUr6r5Xaq0a5GKJvOirPSCcO0fA&oe=69CA3868&_nc_sid=6de079",
    "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_v4axbCdfevl9LMrvUjPLJIeyFRpVsQ2RybsjRcTvp835OfXmPuU1Zze2pBLWZ466DP5OHgh2NNayww-Y1za1cXc38w05KcaUPVrkT8mQUv7pvhxnHTit11cPx8wM8ywmWC9H4IZFzfJIMAw.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adlel2ckvQLkaoKAh2jqef6k5L0A9pgw_lXAEJz8dVIqLesBpD4kb5MkO5kECUHOcdk&ccb=10-5&oh=00_AfvRl2zqF-JRpTbBVLTvknv3K-Zf2Pshs5Jxst_4Xir9bA&oe=69CA18CD&_nc_sid=6de079",
]


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a file with retries."""
    for attempt in range(retries):
        try:
            urlretrieve(url, str(dest))
            return True
        except OSError as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} in {wait}s: {e}")
                import time
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {e}")
                return False
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from predefined URLs (zips and txt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mapillary",
        help="Directory to save the dataset (default: data/mapillary)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, url in enumerate(DATASET_URLS):
            suffix = ".zip" if ".zip?" in url or url.rstrip("/").endswith(".zip") else ".txt"
            path = tmp_dir / f"part_{i:02d}{suffix}"
            print(f"Downloading {i + 1}/{len(DATASET_URLS)}: {path.name} ...")
            if not download_file(url, path):
                print(f"Warning: failed to download {path.name}", file=sys.stderr)
                continue
            if path.suffix.lower() == ".zip":
                print(f"  Extracting to {output_dir} ...")
                with zipfile.ZipFile(path, "r") as zf:
                    for name in zf.namelist():
                        if name.startswith("__MACOSX") or "/." in name:
                            continue
                        zf.extract(name, output_dir)
                path.unlink()
        # Move any .txt from tmp to output root (dataset loader ignores non-dirs)
        for f in tmp_dir.iterdir():
            if f.is_file():
                dest = output_dir / f.name
                if dest.exists():
                    dest.unlink()
                f.rename(dest)
    finally:
        if tmp_dir.exists():
            for f in tmp_dir.iterdir():
                f.unlink()
            tmp_dir.rmdir()

    # If zips extracted with a single top-level dir (e.g. "mapillary/class1/..."), flatten
    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(subdirs) == 1 and subdirs[0].name != "raw_images":
        single = subdirs[0]
        for child in list(single.iterdir()):
            dest = output_dir / child.name
            if dest.exists() and dest.is_dir() and child.is_dir():
                for f in child.iterdir():
                    shutil.move(str(f), str(dest / f.name))
                child.rmdir()
            else:
                if dest.exists():
                    dest.unlink() if dest.is_file() else shutil.rmtree(dest)
                shutil.move(str(child), str(dest))
        single.rmdir()

    print(f"\nDataset directory: {output_dir}")
    total = 0
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            n = len([f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            if n > 0:
                print(f"  {d.name}: {n} images")
                total += n
    print(f"  TOTAL: {total} images")


if __name__ == "__main__":
    main()
