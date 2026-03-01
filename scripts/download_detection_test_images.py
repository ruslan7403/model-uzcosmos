#!/usr/bin/env python3
"""Download ~33 images each for stop sign, traffic light, parking meter from Bing.
Run once: pip install icrawler; python scripts/download_detection_test_images.py
"""
import os

def main():
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("Run: pip install icrawler")
        raise

    root = os.path.join(os.path.dirname(__file__), "..", "data", "detection_test")
    os.makedirs(root, exist_ok=True)

    queries = [
        ("stop_sign", "stop sign on the road", 33),
        ("traffic_light", "traffic light on the road", 33),
        ("parking_meter", "parking meter on the road", 33),
    ]
    for subdir, query, limit in queries:
        out_dir = os.path.join(root, subdir)
        os.makedirs(out_dir, exist_ok=True)
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": out_dir},
        )
        print(f"Downloading up to {limit} images: {query} -> {out_dir}")
        crawler.crawl(keyword=query, max_num=limit)
    print("Done. Images in data/detection_test/")


if __name__ == "__main__":
    main()
