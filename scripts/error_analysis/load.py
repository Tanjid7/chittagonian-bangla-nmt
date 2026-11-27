import pandas as pd, re
MS=['BanglaT5','NLLB-200','mT5-Small','Bangla-BERT2BERT']
R='../../results/'   # run from scripts/error_analysis/
d={m:pd.read_csv(f'{R}preds_finetuned_{m}.csv',keep_default_na=False) for m in MS}
src=d['BanglaT5'].chittagonian.tolist(); ref=d['BanglaT5'].reference.tolist()
P={m:d[m].prediction.tolist() for m in MS}
PUN=re.compile(r'[।॥,;:!?\.\'"“”‘’\-–—()\[\]]')
def tok(s): return PUN.sub(' ',s).split()
