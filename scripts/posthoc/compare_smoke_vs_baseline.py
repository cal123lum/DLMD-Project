#!/usr/bin/env python3
import json, os, re, glob, sys

FAMILY = "bladabindi"
SEED = "42"
BASE_DIRS  = [f"data/processed/metrics/family_final/{FAMILY}", "data/processed/metrics/family_final"]
SMOKE_DIRS = [f"data/processed/metrics/family_smoke/{FAMILY}", "data/processed/metrics/family_smoke"]

frac_re = re.compile(r"_f(?P<frac>0\.\d+)")
seed_re = re.compile(r"_s(?P<seed>\d+)")
variant_re = re.compile(r"_(?P<variant>evogan|gan|evo|smote|oversample|real)\b")

def scan(dirs, must_contain=None):
    out = {}
    for d in dirs:
        for fp in glob.glob(os.path.join(d, "*.json")):
            try:
                j = json.load(open(fp))
            except Exception:
                continue
            tag = j.get("tag") or os.path.basename(fp)
            if must_contain and must_contain not in tag:
                continue
            auc = (j.get("metrics") or {}).get("auc", j.get("auc"))
            if auc is None: 
                continue
            mF = frac_re.search(tag); mS = seed_re.search(tag); mv = variant_re.search(tag)
            frac = mF.group("frac") if mF else f'{j.get("scarce_real_frac")}'
            seed = mS.group("seed") if mS else f'{j.get("seed")}'
            variant = mv.group("variant") if mv else (j.get("variant") or "unknown")
            out[(frac, seed, variant)] = {"auc": float(auc), "path": fp, "tag": tag}
    return out

base = scan(BASE_DIRS, must_contain=FAMILY)
smke = scan(SMOKE_DIRS, must_contain=FAMILY)

rows = []
for (frac, seed, variant), b in base.items():
    if seed != SEED or variant != "evogan":
        continue
    s = smke.get((frac, seed, "evogan"))
    if not s: 
        continue
    rows.append((float(frac), b["auc"], s["auc"], s["auc"]-b["auc"], b["tag"], s["tag"]))

rows.sort(key=lambda r: r[0])
if not rows:
    print("No matching pairs found. Check paths and tags."); sys.exit(1)

print(f"\nAUC comparison (Family {FAMILY}, seed={SEED}):")
print(f"{'frac':>7}  {'base_auc':>8}  {'smoke_auc':>9}  {'Δsmoke-base':>11}")
print("-"*42)
for frac, bauc, sauc, dlt, _, _ in rows:
    print(f"{frac:7.4f}  {bauc:8.4f}  {sauc:9.4f}  {dlt:11.4f}")

def mean(xs): return sum(xs)/len(xs) if xs else 0.0
low = [d for f,_,_,d,_,_ in rows if f in (0.0005, 0.002)]
mid = [d for f,_,_,d,_,_ in rows if f in (0.005, 0.01)]
print("\nSummary:")
print(f"  mean Δ at tiniest (0.0005, 0.002): {mean(low):.4f}")
print(f"  mean Δ at mid (0.005, 0.01):       {mean(mid):.4f}")

if mean(low) >= 0.01 and all(d > -0.01 for d in mid):
    print("\nRecommendation: ✔ RERUN Family with --evo-parent-source both.")
else:
    print("\nRecommendation: ✖ No broad rerun needed; document parent-source per regime.")
