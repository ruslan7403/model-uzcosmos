#!/usr/bin/env python3
"""Download 5-10 traffic sign images from the internet for local testing."""

import os
import sys
import urllib.request

# User-Agent so Wikimedia/servers don't block the request
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"}

# Public domain / free-to-use traffic sign image URLs (direct image links).
# Using Pexels and similar CDNs that allow programmatic download with User-Agent.
TEST_IMAGE_URLS = [
    "https://images.pexels.com/photos/39080/stop-shield-traffic-sign-road-sign-39080.jpeg?auto=compress&cs=tinysrgb&w=640",
    "https://images.pexels.com/photos/10058530/pexels-photo-10058530.jpeg?auto=compress&cs=tinysrgb&w=640",
    "https://images.unsplash.com/photo-1544620246-4b35c2d6d663?w=640",  # road/sign
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=640",  # traffic
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640",  # street
    "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=640",  # city
    "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=640",  # road
    "https://images.unsplash.com/photo-1488190211105-8b0e65b80b4e?w=640",  # street
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "internet_test")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading up to {len(TEST_IMAGE_URLS)} images to {OUTPUT_DIR}")

    for i, url in enumerate(TEST_IMAGE_URLS):
        path = os.path.join(OUTPUT_DIR, f"test_sign_{i + 1:02d}.jpg")
        try:
            req = urllib.request.Request(url, headers=REQUEST_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(path, "wb") as f:
                    f.write(resp.read())
            print(f"  OK: {path}")
        except Exception as e:
            print(f"  FAIL: {url} -> {e}")

    downloaded = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("test_sign_")]
    print(f"\nDownloaded {len(downloaded)} images.")
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(main())
