# ── Reproduces the "Dictionary overlap with the evaluation data" numbers in the paper ──
# Needs the ChatgaiyyaAlap release (for dictionary.csv) and the split files in data/.
# Run from the repository root:  python scripts/dictionary_overlap.py "<path to dictionary.csv>"
import sys, re, unicodedata as u, numpy as np, pandas as pd

PUN = re.compile(r'[।॥,;:!?\.\'"“”‘’\-–—()\[\]]')
tok = lambda s: PUN.sub(' ', u.normalize('NFC', s)).split()
S = {s: pd.read_csv(f'data/{s}_data.csv', keep_default_na=False) for s in ['train', 'val', 'test']}
dic = pd.read_csv(sys.argv[1])
dpairs = set(zip(dic['চট্টগ্রাম'].astype(str).str.strip(), dic['বাংলা'].astype(str).str.strip()))

tr = S['train']
is_dict = np.array([(c, b) in dpairs for c, b in zip(tr.chittagonian, tr.standard_bangla)])
sent, dd = tr[~is_dict], tr[is_dict]
print(f"train: {len(sent)} sentence pairs + {len(dd)} dictionary entries")
D = {tok(c)[0] for c in dd.chittagonian if len(tok(c)) == 1}           # single-word entries
V = {w for c in sent.chittagonian for w in tok(c)}                      # words in training sentences

for name in ['val', 'test']:
    T = [tok(x) for x in S[name].chittagonian]; n = sum(map(len, T))
    a = sum(w in D for s in T for w in s); sa = sum(any(w in D for w in s) for s in T)
    b = sum(w in D and w not in V for s in T for w in s); sb = sum(any(w in D and w not in V for w in s) for s in T)
    print(f"{name}: {100*a/n:.1f}% of tokens in dictionary ({100*sa/len(T):.1f}% of sentences); "
          f"dictionary-only {b} tokens ({100*b/n:.1f}%), {sb} sentences ({100*sb/len(T):.1f}%)")

T = [tok(x) for x in S['test'].chittagonian]
mask = np.array([any(w in D and w not in V for w in s) for s in T])
for m in ['BanglaT5', 'NLLB-200', 'mT5-Small', 'Bangla-BERT2BERT']:
    d = pd.read_csv(f'results/preds_finetuned_{m}.csv', keep_default_na=False)
    em = np.array([u.normalize('NFC', p).strip() == u.normalize('NFC', r).strip() for p, r in zip(d.prediction, d.reference)])
    print(f"{m:<17} exact match: with dictionary-only word {100*em[mask].mean():.1f}% (n={mask.sum()}), others {100*em[~mask].mean():.1f}%")
