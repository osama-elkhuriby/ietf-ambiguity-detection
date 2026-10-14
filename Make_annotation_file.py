"""Split all LLM-annotated rows between 3 annotators. Single self-contained file.

Run:  python make_annotation_files.py --check 5     # print context, verify
      python make_annotation_files.py               # write files

Output (results/annotation/):
  pilot.csv                 -> all 3 annotate first (60 rows), then discuss
  annotator_A.csv, _B, _C   -> each: 240 shared rows + own share of the rest, shuffled together
  key_PRIVATE.csv           -> item_id, record_id, family_id, llm_label, tier, votes_json, group, owner

Text matches what the LLMs saw in annotate_with_ollama.py: document, section_heading,
previous clause, TARGET (clause_text from the dataset), next clause. Same revision, same section.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path("/media/yahya/data_stuff/DATA/Final Project/ietf-ambiguity-detection/corpus/processed")
ANN = BASE / "llm_annotations" / "v1"
ALIGN = BASE / "alignments"
SEED, N_PILOT, N_SHARED = 42, 60, 240
OUT = Path("results/annotation")

_docs = {}


def select_document(family, doc, wanted):
    """Same selection rule as ContextResolver._select_document."""
    if (family, doc) not in _docs:
        best, best_score = [], -1
        for p in sorted((ALIGN / family).glob(f"*/inputs/{doc}.json")):
            try:
                cl = json.loads(p.read_text(encoding="utf-8")).get("clauses")
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(cl, list):
                continue
            score = len(wanted & {c.get("clause_id") for c in cl})
            if score > best_score:
                best, best_score = cl, score
        _docs[(family, doc)] = best if best_score > 0 else []
    return _docs[(family, doc)]


def context_for_row(r):
    """Same logic as ContextResolver._load_context."""
    ids = {c for c in str(r["clause_ids"]).split("|") if c}
    cl = sorted(select_document(r["family_id"], r["document_id"], ids), key=lambda c: int(c.get("index", 0)))
    pos = [i for i, c in enumerate(cl) if c.get("clause_id") in ids]
    if not pos:
        return "", "", ""
    first, last = min(pos), max(pos)
    sec = str(cl[first].get("section_id", ""))
    heading = str(cl[first].get("section_heading", ""))
    prev = nxt = ""
    if first > 0 and str(cl[first - 1].get("section_id", "")) == sec:
        prev = str(cl[first - 1].get("raw_text", ""))
    if last + 1 < len(cl) and str(cl[last + 1].get("section_id", "")) == sec:
        nxt = str(cl[last + 1].get("raw_text", ""))
    return heading, prev, nxt


# ---- load rows (exactly the training dataset the models used) ------------
df = pd.read_csv(ANN / "training_dataset.csv", dtype=str)
cons = pd.read_csv(ANN / "consensus.csv", dtype=str)[["record_id", "votes_json", "candidate_source"]]
df = df.merge(cons, on="record_id", how="left", validate="one_to_one")
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

heads, prevs, nexts = zip(*[context_for_row(r) for r in df.to_dict("records")])
df["section_heading"], df["previous_clause"], df["next_clause"] = heads, prevs, nexts

if "--check" in sys.argv:
    n = int(sys.argv[sys.argv.index("--check") + 1])
    print("rows:", len(df), "| positives:", (df.label == "1").sum())
    for r in df.head(n).to_dict("records"):
        print("DOC    :", r["document_id"], "| HEADING:", r["section_heading"])
        print("PREV   :", r["previous_clause"][:250])
        print("TARGET :", r["clause_text"][:250])
        print("NEXT   :", r["next_clause"][:250], "\n" + "-" * 60)
    sys.exit()

# ---- groups: pilot / shared / individual ---------------------------------
df["item_id"] = range(1, len(df) + 1)
df["group"] = "individual"
df.loc[: N_PILOT - 1, "group"] = "pilot"
df.loc[N_PILOT : N_PILOT + N_SHARED - 1, "group"] = "shared"
df["owner"] = "all"
ind = df.group == "individual"
df.loc[ind, "owner"] = ["ABC"[i % 3] for i in range(ind.sum())]

# ---- write files ---------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["item_id", "document_id", "section_heading", "previous_clause", "clause_text", "next_clause"]


def save(frame, name):
    sheet = frame.sample(frac=1, random_state=SEED)[COLS].rename(
        columns={"document_id": "document", "clause_text": "target_clause"})
    for c in ("ambiguous", "type", "evidence", "confidence"):
        sheet[c] = ""
    sheet.to_csv(OUT / name, index=False, encoding="utf-8-sig")


save(df[df.group == "pilot"], "pilot.csv")
for a in "ABC":
    save(df[(df.group == "shared") | (df.owner == a)], f"annotator_{a}.csv")

df[["item_id", "record_id", "family_id", "label", "llm_label_tier", "votes_json", "candidate_source", "group", "owner"]] \
    .rename(columns={"label": "llm_label"}).to_csv(OUT / "key_PRIVATE.csv", index=False)

print(df.group.value_counts().to_string())
print("rows per annotator:", {a: int(((df.group == "shared") | (df.owner == a)).sum()) for a in "ABC"})