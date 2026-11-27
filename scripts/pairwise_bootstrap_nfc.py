# ── Reproduces the extra numbers in the paper that are NOT in results/ ──
# 1) Paired bootstrap for ALL model pairs (the 99 notebook only tested BanglaT5 vs the rest)
# 2) Exact-match rate per model (NFC)
# 3) Copy-Source baseline (output = Chittagonian input), NFC
# 4) Test-set length statistics
# Same procedure as 99_aggregate_results Cell 3: NFC on both sides, sacreBLEU defaults,
# 1,000 resamples, numpy default_rng(42), one-sided p = (#{b >= a} + 1) / (N + 1).
# It reproduces the notebook's own values exactly (e.g. BanglaT5 vs NLLB-200 BLEU p = 0.1968).
# Run from inside results/:  python ../scripts/pairwise_bootstrap_nfc.py
import unicodedata, itertools, numpy as np, pandas as pd
from sacrebleu.metrics import BLEU, CHRF

NFC = lambda s: unicodedata.normalize("NFC", str(s)).strip()
MODELS = ["BanglaT5", "NLLB-200", "mT5-Small", "Bangla-BERT2BERT"]
N_BOOT, SEED = 1000, 42

P = {}
for m in MODELS:
    d = pd.read_csv(f"preds_finetuned_{m}.csv", keep_default_na=False)
    P[m] = ([NFC(x) for x in d.prediction], [NFC(x) for x in d.reference])
    em = np.mean(np.array(P[m][0]) == np.array(P[m][1])) * 100
    print(f"{m:<17} BLEU {BLEU().corpus_score(P[m][0], [P[m][1]]).score:6.2f} | exact match {em:5.1f}%")

src = [NFC(x) for x in d.chittagonian]
ref = P[MODELS[0]][1]
print(f"Copy-Source       BLEU {BLEU().corpus_score(src, [ref]).score:6.2f} | chrF {CHRF().corpus_score(src, [ref]).score:6.2f}")
for name, col in [("source", src), ("reference", ref)]:
    n = np.array([len(x.split()) for x in col])
    print(f"Test {name}: mean {n.mean():.2f} words, median {np.median(n):.0f}, min {n.min()}, max {n.max()}")

n_test = len(ref)
IDX = np.random.default_rng(SEED).integers(0, n_test, size=(N_BOOT, n_test))
boot = {}
for name, met in [("BLEU", BLEU()), ("chrF", CHRF())]:
    for m in MODELS:
        st = np.array(met._extract_corpus_statistics(P[m][0], [P[m][1]]))
        boot[(m, name)] = np.array([met._compute_score_from_stats(list(st[i].sum(axis=0))).score for i in IDX])

print("\nPaired bootstrap (A better than B?), one-sided")
for a, b in itertools.combinations(MODELS, 2):
    for name in ["BLEU", "chrF"]:
        x, y = boot[(a, name)], boot[(b, name)]
        p = (np.sum(y >= x) + 1) / (N_BOOT + 1)
        lo, hi = np.percentile(x - y, [2.5, 97.5])
        print(f"{a:<10} vs {b:<17} {name:<4} mean diff {np.mean(x - y):6.2f}  "
              f"95% CI of diff [{lo:6.2f}, {hi:6.2f}]  p = {p:.4f}")
