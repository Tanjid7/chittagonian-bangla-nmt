# ── Reproduces the manual error-analysis numbers in the paper ──
# Inter-annotator agreement (Cohen's kappa) between annotator A and B, and per-model
# outcome / error-category counts (from annotator B, as reported in the paper).
# Run from the repository root:  python scripts/annotation_agreement.py
import pandas as pd
from sklearn.metrics import cohen_kappa_score as kappa

rd = lambda p: pd.read_csv(p, keep_default_na=False).set_index("item_id")
A, B = rd("annotation/annotator_A.csv"), rd("annotation/annotator_B.csv").loc[lambda d: d.index]
B = B.loc[A.index]
key = rd("annotation/model_key.csv").loc[A.index]
outcome = lambda s: s.replace({"ref": "ok"})          # 'ref' = output fine, reference wrong -> acceptable

oa, ob = outcome(A.status), outcome(B.status)
print(f"Outcome     n={len(oa)}  agreement {100*(oa == ob).mean():.1f}%  kappa {kappa(oa, ob):.2f}")
both = (oa == "error") & (ob == "error")
ca, cb = A.primary_category[both], B.primary_category[both]
print(f"Category    n={both.sum()}  agreement {100*(ca == cb).mean():.1f}%  kappa {kappa(ca, cb):.2f}")
ra, rb = A.reference_wrong, B.reference_wrong
print(f"Ref. wrong  n={len(ra)}  agreement {100*(ra == rb).mean():.1f}%  kappa {kappa(ra, rb):.2f}")

print("\nOutcome per model (annotator B):")
print(pd.crosstab(ob, key.model))
print("\nMain error category per model (annotator B):")
print(pd.crosstab(B.primary_category[ob == "error"], key.model[ob == "error"]))

print("\nOutputs with each error category, any label (annotator B; paper Table 'manual-cats'):")
rows = [(m, c) for m, cs in zip(key.model, B.categories) for c in set(cs.split("+")) if c]
print(pd.crosstab(pd.Series([c for _, c in rows], name="category"), pd.Series([m for m, _ in rows], name="model")))
