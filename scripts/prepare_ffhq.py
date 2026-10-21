"""Split FFHQ (256x256 PNGs named 00000.png ... 69999.png) into train/ (first 60,000) and test/ (last 10,000).

    python scripts/prepare_ffhq.py --input <ffhq256> --output data/ffhq256
"""

import argparse
import os
import shutil

from tqdm import tqdm

SPLIT_INDEX = 60000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Folder with the FFHQ images (00000.png ...).")
    parser.add_argument("--output", required=True, help="Where to create train/ and test/.")
    args = parser.parse_args()

    for split in ("train", "test"):
        os.makedirs(os.path.join(args.output, split), exist_ok=True)
    for filename in tqdm(sorted(f for f in os.listdir(args.input) if f.endswith(".png"))):
        try:
            idx = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"Skipping non-indexed file: {filename}")
            continue
        split = "train" if idx < SPLIT_INDEX else "test"
        shutil.copy(os.path.join(args.input, filename), os.path.join(args.output, split, filename))


if __name__ == "__main__":
    main()
