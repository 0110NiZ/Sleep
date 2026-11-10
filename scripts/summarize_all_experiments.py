"""Read archived E2-E6 scores; write only to a NEW directory. No inference."""
import argparse, csv, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows=list(csv.DictReader(f))
    table={(int(r["fold"]),r["condition"]):r for r in rows}
    if len(rows)!=50 or len(table)!=50:
        raise ValueError(f"Expected 50 unique fold/condition rows: {path}")
    for r in rows:
        for field in ("macro_f1_pct", "accuracy_pct"):
            value=float(r[field])
            if not 0 <= value <= 100: raise ValueError((path,field,value))
    return table

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e3",type=Path,required=True,help="Formal selected_per_fold.csv, not one fixed filter")
    p.add_argument("--e2",type=Path,default=ROOT/"result/e4_wavelet/e2_reference_per_fold.csv")
    p.add_argument("--e4",type=Path,default=ROOT/"result/e4_wavelet/test_per_fold.csv")
    p.add_argument("--e5",type=Path,default=ROOT/"result/e5_dae/test_per_fold.csv")
    p.add_argument("--e6",type=Path,default=ROOT/"result/e6_noise_augmentation/test_per_fold.csv")
    p.add_argument("--out",type=Path,required=True,help="Must not already exist")
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(f"Refusing to overwrite {a.out}")
    tables={m:read_table(getattr(a,m.lower())) for m in ("E2","E3","E4","E5","E6")}
    names=[c["name"] for c in json.loads((ROOT/"configs/noise_conditions_v2.json").read_text())["conditions"]]
    expected={(f,c) for f in range(5) for c in names}
    for m,t in tables.items():
        if set(t)!=expected: raise ValueError(f"Unexpected folds/conditions: {m}")
    overall=[];paired=[]
    for m,t in tables.items():
        clean=[float(t[f,"Clean"]["macro_f1_pct"]) for f in range(5)]
        noisy=[statistics.mean(float(t[f,c]["macro_f1_pct"]) for c in names if c!="Clean") for f in range(5)]
        overall.append(dict(method=m,noisy_mean_macro_f1_pct=statistics.mean(noisy),noisy_sd_macro_f1_pct=statistics.stdev(noisy),clean_mean_macro_f1_pct=statistics.mean(clean),clean_sd_macro_f1_pct=statistics.stdev(clean)))
        for (f,c),r in sorted(t.items()):
            paired.append(dict(method=m,fold=f,condition=c,macro_f1_pct=float(r["macro_f1_pct"]),delta_vs_e2_pp=float(r["macro_f1_pct"])-float(tables["E2"][f,c]["macro_f1_pct"])))
    a.out.mkdir(parents=True,exist_ok=False)
    for filename,rows in (("overall.csv",overall),("paired_vs_e2.csv",paired)):
        with (a.out/filename).open("x",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out/"README.txt").write_text("Descriptive five-fold comparisons; percentages and percentage-point differences. SD is sample SD of five per-fold values. Folds are not independent repetitions. Timing is excluded because recorded timing protocols differ. Archived source files are unchanged.\n",encoding="utf-8")
    print(a.out)
if __name__=="__main__": main()
