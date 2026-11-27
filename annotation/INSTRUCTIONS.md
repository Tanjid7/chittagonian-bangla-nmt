# Instructions for annotators (A and B)

Work **alone**. Do not look at the other annotator's file. Fill only the empty columns.

| Column | What to write |
|---|---|
| `status` | `exact` (same as reference) · `ok` (different words, same meaning) · `error` (translation is wrong) · `ref` (the output is fine but the reference is wrong) |
| `categories` | Only if `error`: one or more codes joined with `+`, e.g. `LEX+OMI` |
| `primary_category` | The most serious code from `categories` |
| `reference_wrong` | `yes` if the reference itself is wrong or adds/removes meaning, otherwise `no` |
| `note` | Optional short comment |

Codes (use only these):
OMI = a word is missing · ADD = extra word not in the source ·
LEX = wrong word / wrong meaning · MOR = right word, wrong ending (tense, person, case) ·
UNT = Chittagonian word left untranslated · OTH = any other problem (negation, word order, repetition, unrelated output)

Each source sentence appears 4 times with different outputs, in random order. Judge each output on its own.
Save as CSV (UTF-8) and send both files back.
