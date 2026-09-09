"""Compare two complete binary files, preserving signed-zero and NaN payload differences."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();expected=a.reference.read_bytes();actual=a.candidate.read_bytes()
    mismatch=sum(x!=y for x,y in zip(expected,actual))+abs(len(expected)-len(actual))
    print(json.dumps(dict(equal=expected==actual,reference_bytes=len(expected),candidate_bytes=len(actual),
        differing_bytes=mismatch,reference_sha256=hashlib.sha256(expected).hexdigest(),
        candidate_sha256=hashlib.sha256(actual).hexdigest()),indent=2))
    if expected!=actual:raise SystemExit(1)

if __name__=='__main__':main()
