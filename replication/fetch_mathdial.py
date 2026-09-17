"""Fetch the public MathDial release (not redistributed with this package).

MathDial (Macina et al., 2023) is distributed by its authors at
https://github.com/eth-nlped/mathdial under that repository's licence. This script
downloads the two split files into ./mathdial/ so the replication notebook and the
human-verification sheet can be rebuilt.
"""
import os, urllib.request

BASE = "https://raw.githubusercontent.com/eth-nlped/mathdial/main/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mathdial")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        dst = os.path.join(OUT, f"{split}.csv")
        urllib.request.urlretrieve(f"{BASE}/{split}.csv", dst)
        print(f"{split}.csv  {os.path.getsize(dst):,} bytes  ->  {dst}")
