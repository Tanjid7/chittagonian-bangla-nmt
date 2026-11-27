# Chittagonian → Standard Bangla Machine Translation Benchmark

Code, predictions and annotation files for the paper
**"TODO: paper title"** (TODO: authors, venue, year).

We fine-tune and compare four pretrained models (mT5-Small, BanglaT5, NLLB-200-distilled-600M,
Bangla-BERT2BERT) on the same frozen train / validation / test split, with the same decoding
(beam 4, max length 64) and the same metrics (sacreBLEU, chrF, ROUGE-L, BERTScore), after
Unicode NFC normalisation.

## Repository layout

| Folder | Contents |
|---|---|
| `notebooks/` | Kaggle notebooks, run in this order: `00_data_prep` → `01`–`04` (one model each) → `99_aggregate_results` |
| `data/` | The frozen train / validation / test split used in all experiments (+ `data_manifest.json`) |
| `results/` | Test-set predictions (`preds_*.csv`), scores, training-loss histories and run records (`run_config_*.json`: hyperparameters, library versions, split MD5s) |
| `results/nfc_rescored/` | Scores after NFC normalisation (the numbers reported in the paper) and old-vs-NFC comparison |
| `scripts/` | Extra analyses reported in the paper (see below) |
| `annotation/` | Manual error-analysis labels from two annotators (100 test sentences × 4 models = 400 outputs) |

## Data

The three source datasets are released on Mendeley Data under CC BY 4.0. The raw releases are
not copied here; download them from their original pages and cite them. The frozen split
derived from them is in `data/` (same licence, with attribution to the sources below):

1. **ChatgaiyyaAlap**: https://doi.org/10.17632/wtms9xbkkw.1
2. **Chattogram Multilingual Sentiment Dataset**: https://doi.org/10.17632/k6hts2ktxw.1
3. **ANCHOLIK-NER**: https://doi.org/10.17632/gbkszkt8z3.4

`00_data_prep` builds the split (grouped by the Standard Bangla sentence, `random_state=42`) and
writes `data_manifest.json` with an MD5 fingerprint of each split. Notebooks 01–04 stop if the
fingerprint does not match. Running `00_data_prep` on the current public releases reproduces
the three files in `data/` byte for byte:

| Split | Pairs | MD5 |
|---|---|---|
| train | 6,924 (5,422 sentences + 1,502 dictionary entries) | `37a8c7f1d8fbf214ea419aef2f068ec4` |
| val | 1,169 | `3a4467f766b93dab1afe1a4dcb775b47` |
| test | 1,162 | `a523044309350456dad01b182d27d162` |

## How to reproduce

1. Run `notebooks/00_data_prep.ipynb` on Kaggle (CPU). Save its output files as a Kaggle dataset.
2. Run `01`–`04` on a single T4 GPU, one notebook per session (Internet ON).
3. Run `99_aggregate_results.ipynb` (CPU) with the outputs of 01–04 as input. It re-scores
   all saved predictions with NFC normalisation and computes bootstrap CIs and significance.
   It never retrains a model.

Paths in the notebooks point to Kaggle inputs (`/kaggle/input/...`); change them if you use your
own Kaggle account or another platform.

## Extra scripts (no GPU needed)

```bash
pip install sacrebleu pandas numpy scikit-learn

# all-pairs paired bootstrap (one- and two-sided p), exact match, copy-source baseline, test length stats
cd results && python ../scripts/pairwise_bootstrap_nfc.py && cd ..

# inter-annotator agreement (Cohen's kappa) and per-model error counts
python scripts/annotation_agreement.py

# dictionary words in val/test inputs (needs ChatgaiyyaAlap's dictionary.csv)
python scripts/dictionary_overlap.py "<path>/dictionary.csv"

# check that every test pair comes from the three public sources (needs the raw datasets)
python scripts/data_audit.py

# automatic error analysis (edit profile, length buckets, NFC effect)
cd scripts/error_analysis && python auto2.py && python auto3.py && python nfc.py
```

## Annotation files

`annotation/annotator_A.csv` and `annotator_B.csv` hold the labels; `model_key.csv` maps each
`item_id` to the model and test row (annotators did not see it). Codes are explained in
`annotation/INSTRUCTIONS.md`. Annotator A labelled from a blank sheet. Annotator B reviewed
AI-assisted first-pass labels.

## Citation

```bibtex
TODO: add BibTeX of the paper once published
```

## License

TODO: choose a license for the code (e.g. MIT). The datasets keep their original licenses.
