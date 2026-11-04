import argparse, os, random
import pandas as pd
from pyjaspar import jaspardb

jdb = jaspardb(release="JASPAR2024")
MOTIF_IDS = {"FOSL2": "MA0478.1", "CTCF": "MA0139.1"}   # confirm IDs exist (see check below)

def load_pfm(name):
    m = jdb.fetch_motif_by_id(MOTIF_IDS[name])
    probs = m.counts.normalize(pseudocounts=0.5)
    return [[probs[b][i] for b in "ACGT"] for i in range(len(m))]

def rand_seq(L, gc, rng):
    w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]    # A, C, G, T
    return "".join(rng.choices("ACGT", weights=w, k=L))

def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))

def sample_site(pfm, rng):
    return "".join(rng.choices("ACGT", weights=w, k=1)[0] for w in pfm)

def mutate(s, rate, rng):
    return "".join(rng.choice([b for b in "ACGT" if b != c]) if rng.random() < rate else c
                   for c in s)

def plant(bg, pfm, rate, rng):
    m = mutate(sample_site(pfm, rng), rate, rng)
    if rng.random() < 0.5:
        m = revcomp(m)
    p = rng.randint(0, len(bg) - len(m))
    return bg[:p] + m + bg[p + len(m):]

def shuffle(s, rng):
    l = list(s); rng.shuffle(l); return "".join(l)

ap = argparse.ArgumentParser()
ap.add_argument("--motif", default="FOSL2", choices=list(MOTIF_IDS))
ap.add_argument("--rate", type=float, default=0.1)
ap.add_argument("--neg", default="shuffled", choices=["shuffled", "background"])
ap.add_argument("--n_train", type=int, required=True)    # number of PAIRS
ap.add_argument("--n_val", type=int, default=None)
ap.add_argument("--n_test", type=int, required=True)     # number of PAIRS
ap.add_argument("--gc", type=float, default=0.5)
ap.add_argument("--L", type=int, default=101)
ap.add_argument("--seed", type=int, default=41)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(a.seed)
pfm = load_pfm(a.motif)
n_val = a.n_val if a.n_val is not None else max(1, round(0.1 * a.n_train))
total = a.n_train + n_val + a.n_test

pairs = []
for _ in range(total):
    pos = plant(rand_seq(a.L, a.gc, rng), pfm, a.rate, rng)
    neg = shuffle(pos, rng) if a.neg == "shuffled" else rand_seq(a.L, a.gc, rng)
    pairs.append((pos, neg))

rng.shuffle(pairs)
train = pairs[:a.n_train]
val = pairs[a.n_train:a.n_train + n_val]
test = pairs[a.n_train + n_val:]

os.makedirs(a.out, exist_ok=True)
for name, prs in [("train", train), ("val", val), ("test", test)]:
    rows = [(p, 1) for p, _ in prs] + [(n, 0) for _, n in prs]
    pd.DataFrame(rows, columns=["sequence", "label"]).sample(
        frac=1, random_state=a.seed).to_csv(f"{a.out}/{name}.csv", index=False)
print("done:", a.out, "| train/val/test pairs:", a.n_train, n_val, a.n_test)
