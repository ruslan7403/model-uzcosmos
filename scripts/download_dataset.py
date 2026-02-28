#!/usr/bin/env python3
"""Download traffic sign dataset from pre-hosted file links.

Downloads ZIP archives and metadata files from direct URLs,
extracts them, and organizes images into a class-per-directory
structure suitable for training.

No API token required — uses direct download links.

Usage:
    python scripts/download_dataset.py --output-dir data/mapillary
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Direct download URLs for the traffic sign dataset.
#
# ZIP files contain traffic sign images (possibly organized by class).
# TXT files contain metadata / class mappings.
#
# To update these links, replace the URLs below.
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "metadata": [
        {
            "name": "metadata_1.txt",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8RSqTVAOhQUZHE5NgIL-tgliMqFe4h3ieVXSDNI-ilkh922n_TtB2VpiW-SzAHEYLiqk3Y3FU8pM-DskkOMCvFGh9RJAxxqi3RI2JXDvbQfq4xYwVstPYKlT44GhxzbDbdW_OG0lyY.txt?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adkf0QW3jQuJg9vRQudIFxERexeIlNJWnlY6LnEYKDQv7JHNJhZ_D8lsJqaNZh3O-B8&ccb=10-5&oh=00_AfsXaLhmgowJY69KQlQfCH79bMCXM1YceO__UB87MvHG8A&oe=69CA1D45&_nc_sid=6de079",
        },
        {
            "name": "metadata_2.txt",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9FlyymG8PIT47PpdF-es5LyB3G1VvhyKar744ioBXCcFVQDaTVBq1LTvq1vz61u49tefGZb03n-mQ_PWKPIX8ZrMfRI09WMTygDexYMZJ6VkTWJK_7FTY47dC9KeKQG8RG3oGU6Dhw.txt?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlmzuPRWIcDay_D4Kgg24mq_6HQLGB57pQzAtggBbvt_FtqogZlXP7JzCCBuX-zrt0&ccb=10-5&oh=00_AfuYJ1FS0mDr7oNLmLdoYlEg-kWzxdJ1ySlZmYXa9j85uA&oe=69CA2EC3&_nc_sid=6de079",
        },
    ],
    "archives": [
        {
            "name": "archive_01.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An90x89nHvauCK1fqMJ8110KeTjNo5Si7rzhvwIMCu5xI9_GhWBGOIXaFvu6o53NuNpBMzdC9qsjAVR8sLv8m6WoFfn6Qd4NjMYKNW4NCKVp6gx3MhZtwf3cZR94wFhou5lPI0hGUw.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdmnIHbGPbvbfA4T6e2RZ2ljiljqeDAi_NuMctzziXbGNEs6q4V26tiXUdYsI9zl71I&ccb=10-5&oh=00_Afsex7q0w02AScH0fhLzl54ZKNoQ63wlPLN4cSODsLSbCA&oe=69CA0E88&_nc_sid=6de079",
        },
        {
            "name": "archive_02.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9eB8zXYW473CiQYW9CPvGx1Ho-fkQvkini3ddExpFOz47aWs4ydBSvK-ZhOPu7ikASQmZvX0zyXhmJzBr6CDZE5ZkUhvJ44h7mV2NT4cSRbR837J9mHJosreQRJJdGaVDR26EAjLPL.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdnlDs4vn35vQu5MVm8KE620i9f23HfcgWxYXMFS658b1F6xv4c3wL2QvSrrro_v9bY&ccb=10-5&oh=00_AftAgfJ0pycNuJxTWr-y_bLjgXBecrkoOhb04pVhxAc2Jw&oe=69CA181C&_nc_sid=6de079",
        },
        {
            "name": "archive_03.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_WKGcw-ICowA_xAgTEU_E-pAYdybyzT-9Pwi8JtelanWnNRKONV1DTAZPEAsGNDWlFpYDi16km1stDN47ip-quE77cfkv3aERdMIRahysGgspb6DlCgrabPSTFJI3tZ9EMRatRC6ZmjQytVcY.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlhNFHfVcTLlCmkFo1lcDRyjjEmPSutbfma5a5lfxnO-Yx065sy5mj7NVYVx1-cw6Y&ccb=10-5&oh=00_Afv_rbOWR_itZKG9cys_6waZ5FNQ0ZMifIPo99L_Kau_8Q&oe=69CA1472&_nc_sid=6de079",
        },
        {
            "name": "archive_04.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8VtaI-ldaSOc5HcLFVQ6SPvHDt50hLG1kga0nUfswldLu1J9dsOx6ynZicRUuXR_TvsczpplOqQEa7ppT4JwUzI0ZNQCHmhtkfT5tjdNJY55Ud6eXplvq59PjOx55d2EbIxYpO9vhR-BcflQ.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdkzDWnefJ-Db_muEXHMUM4TzFwP64wozlaor3oJLZLMXZT4BMd4ICL3TtkdsCNTgkU&ccb=10-5&oh=00_AfvfoxCPHSLe4U0PWSD-iCb97aXxFpg-1-YkEC9QTTHwLA&oe=69CA265E&_nc_sid=6de079",
        },
        {
            "name": "archive_05.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9BwhO8zTAy6jGrwy71BjIBtMK8K5RkpIJguP7DVnpJK2TfKDlfxXj8mCxRJss4zzfaaKi2idqbQOtYJ740TPCI7w7hL8V7goknzuO0ZFPLywDCKIB7i64lCiSUNYXLqeS8mC7EkiU5hYAfOcI.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlrNTk2ctb2KXEUiaw9oVBaVZwuLkqQgRrPx7YVXtKVa8NoVWryXiNj2c16KtT6KT8&ccb=10-6&oh=00_AfurDewpgTt5Biy45uuR9oxb7hmsJMWX6O1OqOJ0pjkLjA&oe=69CA0FDB&_nc_sid=6de079",
        },
        {
            "name": "archive_06.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An9tl3SwDwRFQ2z9tAag26brYamdQZmHnVoxfNTz_Iass-zZLWM-HryqW44UeqbLWd-EkXVIP-ZQQfg3F7dmQYlnu1wjzCARviaJMBHgtLH4gTAeW6msFbEXA3_NIZBtdP7Gg8dt5Ewl.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdlW-xoWKeve6GWxjBw22h9Yddkjd5Fsnro6dNyPiS7GyllE8wIgeJkD1Py4dIPTxDY&ccb=10-5&oh=00_AfvabCHO64OhX5tU27xS6C2aczj2HRiFbphpgHkOyEDJzA&oe=69CA2F67&_nc_sid=6de079",
        },
        {
            "name": "archive_07.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8dmcHh74W4QOmBjSUnIgjyIDl7pVOj8ym708qotPkG4w_03qYL9KUrHpljvhZgYfNJp82zqzc5ZNAawW6W-huIX9hXtE0Q1wEbbbJWyP2Qgfosi_118TvAPBpKxis6XM8RroqJwg.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adlbe1ae0reUeGDmAFyIFd1zitVuvjjenMio3n-XdXD09HPfKKY4GT6iBZUjBJR6XnU&ccb=10-5&oh=00_AfvOkp1p6sF1VNkdakpCVGwIu_BtFNzcLiA6_RA5gEKSxw&oe=69CA3550&_nc_sid=6de079",
        },
        {
            "name": "archive_08.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_XCkEh3XjG7bV8c4h5QX-Qodh85URfwXB83u_txLJaEcDmFH6VOUG0O2T8U5WZt7mpEa-EczZiupLH93KsSXfDAIPq8HjPJn1q_oRSBA3ufZus7MnJ_CGs1O166IJiczbnq9J-0NOZ_G5mR8c.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adn0F6YkwqRkAKZXG-F9MXlclEBmpnrBAxFey4wz4WWOtftacrdHmDk455my2XdnLd8&ccb=10-5&oh=00_Afs6T6FcCK0NRloM4pD9wozCJWvS7ScJI6G_cEMb6pZKyg&oe=69CA18C0&_nc_sid=6de079",
        },
        {
            "name": "archive_09.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An8bwxH70h7HKMm7H2IGV4ptYgE55ozIcf4_wWboduU4ToWxKTnWIelPbSqQ3C1RxKoRTjsOfB9_gzdrv6Oq6qLxEJ1IfOuOXvv6btXEwLfAw2feJOla96rwgmmQyhf-AzgqpS1yIKeydT8zkA.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adm5rfWythwb17S-ZPg_IgaLdFoxdIZCNTx0EKSZ4Zc8Gzvq6Q5p65qR7rGed1cKynE&ccb=10-5&oh=00_AftvVpRy1PYyygjqmo9Rc7SThX6axV_eZkBLYy41TdSeDw&oe=69CA228B&_nc_sid=6de079",
        },
        {
            "name": "archive_10.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An-nQsgt_uWGakflQM3LbwL32CaQBwAqrR1r2jdVCHNoR34x1v6LHlvlpTQ9CM86r3tREP05tP049I58J7utUv4vefGA3XD2Up-fJ5cXubNGDCglkw-haZwHnvQR-QtqOisk_5IOaEUK62MYBAE.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=AdmsNfWK4XeuoxPyZAWOJ_kgxOMIEIUKSnHDxz7CHFgKNJBqZQC2OJ6cVuDlsou_Lj4&ccb=10-5&oh=00_AfshK6BNGusfB4D9ImWuUr6r5Xaq0a5GKJvOirPSCcO0fA&oe=69CA3868&_nc_sid=6de079",
        },
        {
            "name": "archive_11.zip",
            "url": "https://scontent.ftas1-2.fna.fbcdn.net/m1/v/t6/An_v4axbCdfevl9LMrvUjPLJIeyFRpVsQ2RybsjRcTvp835OfXmPuU1Zze2pBLWZ466DP5OHgh2NNayww-Y1za1cXc38w05KcaUPVrkT8mQUv7pvhxnHTit11cPx8wM8ywmWC9H4IZFzfJIMAw.zip?_nc_gid=VD5CXbM5qAf_2pQtHXcP2A&_nc_oc=Adlel2ckvQLkaoKAh2jqef6k5L0A9pgw_lXAEJz8dVIqLesBpD4kb5MkO5kECUHOcdk&ccb=10-5&oh=00_AfvRl2zqF-JRpTbBVLTvknv3K-Zf2Pshs5Jxst_4Xir9bA&oe=69CA18CD&_nc_sid=6de079",
        },
    ],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


def download_file(url: str, dest: str, retries: int = 4) -> bool:
    """Download a file with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            print(f"  Downloading {os.path.basename(dest)}...")
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  Done ({size_mb:.1f} MB)")
            return True
        except (urllib.error.URLError, OSError) as e:
            wait = 2 ** (attempt + 1)
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  Failed to extract {zip_path}: {e}")
        return False


def is_image(path: str) -> bool:
    """Check if a file is an image based on extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def discover_and_organize(extract_dir: str, output_dir: str) -> int:
    """Discover images from extracted archives and organize into class dirs.

    Handles multiple structures:
    1. ZIP contains class subdirectories with images -> use directory names
    2. ZIP contains a single directory with class subdirs -> use those
    3. ZIP contains flat images -> group by filename prefix or parent dir name
    4. Nested directories -> walk and use deepest directory as class name

    Returns the number of images organized.
    """
    os.makedirs(output_dir, exist_ok=True)
    organized = 0

    # Walk through all extracted content
    class_images: dict[str, list[str]] = {}

    for root, dirs, files in os.walk(extract_dir):
        # Skip __MACOSX and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]

        image_files = [f for f in files if is_image(f) and not f.startswith(".")]
        if not image_files:
            continue

        # Determine class name from directory structure
        rel_path = os.path.relpath(root, extract_dir)
        parts = Path(rel_path).parts

        if rel_path == ".":
            # Images at root level — use filename prefix as class
            for img in image_files:
                stem = Path(img).stem
                # Try to extract class from filename patterns like
                # "stop_001.jpg", "speed_limit_30_002.jpg"
                # Use everything before the last underscore+digits
                class_name = _class_from_filename(stem)
                class_images.setdefault(class_name, []).append(
                    os.path.join(root, img)
                )
        else:
            # Use the most meaningful directory name as class
            # If structure is like: archive_01/stop_sign/img.jpg -> "stop_sign"
            # If structure is like: train/stop_sign/img.jpg -> "stop_sign"
            # If single dir deep: stop_sign/img.jpg -> "stop_sign"
            class_name = _class_from_path(parts)
            for img in image_files:
                class_images.setdefault(class_name, []).append(
                    os.path.join(root, img)
                )

    # Copy images to output class directories
    for class_name, images in sorted(class_images.items()):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for src_path in images:
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(class_dir, dst_name)

            # Handle duplicates by adding a suffix
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(dst_name)
                dst_path = os.path.join(class_dir, f"{name}_{counter}{ext}")
                counter += 1

            shutil.copy2(src_path, dst_path)
            organized += 1

    return organized


def _class_from_filename(stem: str) -> str:
    """Extract a class name from a filename stem.

    Handles patterns like:
        "stop_001" -> "stop"
        "speed_limit_30_002" -> "speed_limit_30"
        "img_0001" -> "unknown"
    """
    import re
    # Remove trailing _NNN or _NNNN numeric suffix
    cleaned = re.sub(r"[_-]\d{1,5}$", "", stem)
    # Remove just trailing digits
    cleaned = re.sub(r"\d+$", "", cleaned).rstrip("_- ")
    return cleaned if cleaned else "unknown"


def _class_from_path(parts: tuple) -> str:
    """Determine the best class name from a relative path.

    Skips generic top-level names like 'train', 'test', 'images', 'data'.
    """
    generic = {"train", "test", "val", "valid", "validation", "images",
               "data", "dataset", "raw", "raw_images", "extracted"}

    # Walk from deepest to shallowest, pick first non-generic name
    for part in reversed(parts):
        if part.lower() not in generic:
            return part

    # All parts are generic — use the full path joined
    return "_".join(parts)


def print_summary(output_dir: str):
    """Print a summary of the organized dataset."""
    print("\n=== Dataset Summary ===")
    total = 0
    classes = 0
    for d in sorted(os.listdir(output_dir)):
        dp = os.path.join(output_dir, d)
        if os.path.isdir(dp):
            n = len([f for f in os.listdir(dp) if is_image(f)])
            if n > 0:
                print(f"  {d}: {n} images")
                total += n
                classes += 1
    print(f"\n  Classes: {classes}")
    print(f"  Total images: {total}")

    if classes < 2:
        print("\nWARNING: Need at least 2 classes with >= 2 images each for training!")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download traffic sign dataset from direct links"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mapillary",
        help="Directory to save the organized dataset (default: data/mapillary)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary download/extract directory for debugging",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Skip if dataset already present (e.g. after Railway redeploy with same volume)
    existing_classes = 0
    if os.path.isdir(output_dir):
        for sub in Path(output_dir).iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                n_images = sum(1 for f in sub.rglob("*") if f.is_file() and is_image(f.name))
                if n_images > 0:
                    existing_classes += 1
    if existing_classes >= 3:
        print(f"Dataset already present at {output_dir} ({existing_classes} class dirs with images). Skipping download.")
        print("To re-download, remove or rename the output directory and run again.")
        return

    # Create temp directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="traffic_signs_")
    download_dir = os.path.join(temp_dir, "downloads")
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(download_dir)
    os.makedirs(extract_dir)

    print(f"Temp directory: {temp_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Download metadata files
        print("\n--- Downloading metadata files ---")
        for meta in DATASET_FILES["metadata"]:
            dest = os.path.join(download_dir, meta["name"])
            if download_file(meta["url"], dest):
                # Copy metadata to output dir for reference
                shutil.copy2(dest, os.path.join(output_dir, meta["name"]))

        # Download and extract archives
        print("\n--- Downloading dataset archives ---")
        total_archives = len(DATASET_FILES["archives"])
        for i, archive in enumerate(DATASET_FILES["archives"], 1):
            print(f"\n[{i}/{total_archives}] {archive['name']}")
            dest = os.path.join(download_dir, archive["name"])

            if not download_file(archive["url"], dest):
                print(f"  Skipping {archive['name']} (download failed)")
                continue

            # Extract to a per-archive subdirectory
            archive_extract_dir = os.path.join(
                extract_dir, Path(archive["name"]).stem
            )
            os.makedirs(archive_extract_dir)

            print(f"  Extracting...")
            if extract_zip(dest, archive_extract_dir):
                n_files = sum(
                    1 for _, _, files in os.walk(archive_extract_dir)
                    for f in files if is_image(f)
                )
                print(f"  Found {n_files} images")
            else:
                print(f"  Extraction failed, skipping")

            # Remove the zip to free disk space
            os.remove(dest)

        # Organize all extracted images into class directories
        print("\n--- Organizing images into class directories ---")
        total_organized = discover_and_organize(extract_dir, output_dir)
        print(f"\nOrganized {total_organized} images total")

        # Print summary
        ok = print_summary(output_dir)

        if not ok:
            print("\nDataset may not have enough classes.")
            print("Check the archive contents and update the organization logic.")
            sys.exit(1)

    finally:
        if args.keep_temp:
            print(f"\nTemp directory preserved: {temp_dir}")
        else:
            print(f"\nCleaning up temp directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
