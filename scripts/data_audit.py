# ── Reproduces the source counts in Section 3 / Table "Source datasets" ──
# Needs the three public releases unzipped in the current folder (same folder
# names as downloaded) and results/ for the frozen test set.
# Cleaning = the same `sanitize` steps as 00_data_prep.ipynb.
import pandas as pd, collections, sys
C = 'ChatgaiyyaAlap A Dataset for Conversion from Chittagonian Dialect to Standard Bangla/'
SRC = {
 'ChatgaiyyaAlap': (C + 'Dataset_Chittagong_2.0.csv', {'বাংলা': 'standard_bangla', 'চট্টগ্রাম': 'chittagonian'}),
 'ANCHOLIK-NER':   ('ANCHOLIK-NER A Benchmark Dataset for Bangla Region/Regional_NER (Raw Sentences).csv',
                    {'Standard_Bangla': 'standard_bangla', 'Chittagong': 'chittagonian'}),
 'Chattogram sent': ('Chattogram_Multilingual_Sentiment_Dataset/chattogram_Bengali_English_language_sentiment_analysis.csv',
                    {'Bengali': 'standard_bangla', 'Chattogram': 'chittagonian'}),
}
DICT = (C + 'dictionary.csv', {'বাংলা': 'standard_bangla', 'চট্টগ্রাম': 'chittagonian'})
TEST = sys.argv[1] if len(sys.argv) > 1 else 'results/preds_finetuned_BanglaT5.csv'

def sanitize(df, cols):
    df = df[list(cols)].rename(columns=cols).copy().dropna(subset=list(cols.values()))
    for c in cols.values():
        df[c] = df[c].astype(str); df = df[df[c] != 'nan']
        df[c] = df[c].str.strip(); df = df[df[c].str.len() > 0]
    return df.drop_duplicates().reset_index(drop=True)

clean = {}
for k, (f, cols) in SRC.items():
    raw = pd.read_csv(f); clean[k] = sanitize(raw, cols)
    print(f"{k:<16} rows {len(raw):>5} | cleaned {len(clean[k]):>5}")
dic = sanitize(pd.read_csv(DICT[0]), DICT[1]); print(f"dictionary       rows {len(pd.read_csv(DICT[0])):>5} | cleaned {len(dic):>5}")

P = {k: set(zip(v.chittagonian, v.standard_bangla)) for k, v in clean.items()}
ks = list(P)
for i in range(3):
    for j in range(i + 1, 3):
        print(f"shared pairs {ks[i]} & {ks[j]}: {len(P[ks[i]] & P[ks[j]])}")
print("unique pairs in rebuild:", len(set().union(*P.values())))

t = pd.read_csv(TEST, keep_default_na=False)
tp = set(zip(t.chittagonian, t.reference))
for k in P: print(f"test pairs found in {k}: {len(tp & P[k])}")
print("test pairs found in any source:", len(tp & set().union(*P.values())), "of", len(tp))
print(collections.Counter(tuple(sorted(k for k in P if p in P[k])) for p in tp))
corpus_ctg = set().union(*[set(v.chittagonian) for v in clean.values()])
print("dictionary entries whose Chittagonian matches a corpus sentence:", dic.chittagonian.isin(corpus_ctg).sum())
