"""
アンカー情報だけでテストデータのカラムをどこまで埋められるか検証する。
"""
import json, csv, os

ANCHOR_DIR = "pipeline/pride_anchors"
SAMPLE_SUB = "SampleSubmission.csv"

# Modification MOD → UNIMOD 変換テーブル (頻出のみ)
MOD_TO_UNIMOD = {
    "MOD:00397": {"name": "Carbamidomethyl", "unimod": "UNIMOD:4", "ta": "C", "mt": "Fixed",
                   "sdrf": "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed"},  # iodoacetamide
    "MOD:00425": {"name": "Oxidation", "unimod": "UNIMOD:35", "ta": "M", "mt": "Variable",
                   "sdrf": "NT=Oxidation;AC=UNIMOD:35;TA=M;MT=Variable"},  # monohydroxylated
    "MOD:00394": {"name": "Acetyl", "unimod": "UNIMOD:1", "ta": "", "mt": "Variable",
                   "sdrf": "NT=Acetyl;AC=UNIMOD:1;PP=Protein N-term;MT=Variable"},  # acetylated
    "MOD:00696": {"name": "Phospho", "unimod": "UNIMOD:21", "ta": "S,T,Y", "mt": "Variable",
                   "sdrf": "NT=Phospho;AC=UNIMOD:21;TA=S,T,Y;MT=Variable"},  # phosphorylated
    "MOD:00400": {"name": "Deamidated", "unimod": "UNIMOD:7", "ta": "N,Q", "mt": "Variable",
                   "sdrf": "NT=Deamidated;AC=UNIMOD:7;TA=N,Q;MT=Variable"},  # deamidated
    "MOD:01148": {"name": "GlyGly", "unimod": "UNIMOD:121", "ta": "K", "mt": "Variable",
                   "sdrf": "NT=GlyGly;AC=UNIMOD:121;TA=K;MT=Variable"},  # ubiquitinylated
}

# IAA → AlkylationReagent + ReductionReagent 推論
ALKYLATION_FROM_MOD = {
    "MOD:00397": "iodoacetamide",  # iodoacetamide derivatized residue
}

# Load SampleSubmission to get test rows
with open(SAMPLE_SUB) as f:
    reader = csv.DictReader(f)
    sub_rows = list(reader)
    columns = reader.fieldnames

# Group by PXD
from collections import defaultdict
pxd_rows = defaultdict(list)
for row in sub_rows:
    pxd_rows[row["PXD"]].append(row)

# Stats
total_cells = 0
filled_cells = 0
filled_by_col = defaultdict(int)
total_by_col = defaultdict(int)

for pxd, rows in sorted(pxd_rows.items()):
    anchor_file = os.path.join(ANCHOR_DIR, f"{pxd}_anchor.json")
    if not os.path.exists(anchor_file):
        continue
    
    with open(anchor_file) as f:
        anchor = json.load(f)
    
    for row in rows:
        # Organism
        if anchor["organism"]:
            org = anchor["organism"][0]
            row["Characteristics[Organism]"] = org["name"].lower()
            filled_by_col["Characteristics[Organism]"] += 1
        
        # Instrument
        if anchor["instruments"]:
            inst = anchor["instruments"][0]
            row["Comment[Instrument]"] = inst["sdrf_format"]
            filled_by_col["Comment[Instrument]"] += 1
        
        # FragmentationMethod (from lookup)
        if anchor["inferred_fragmentation"]:
            row["Comment[FragmentationMethod]"] = anchor["inferred_fragmentation"]
            filled_by_col["Comment[FragmentationMethod]"] += 1
        
        # MS2MassAnalyzer
        if anchor["inferred_ms2_analyzer"]:
            row["Comment[MS2MassAnalyzer]"] = anchor["inferred_ms2_analyzer"]
            filled_by_col["Comment[MS2MassAnalyzer]"] += 1
        
        # IonizationType
        if anchor["inferred_ionization"]:
            row["Comment[IonizationType]"] = anchor["inferred_ionization"]
            filled_by_col["Comment[IonizationType]"] += 1
        
        # OrganismPart
        if anchor["organism_parts"]:
            parts = [op["name"].lower() for op in anchor["organism_parts"] if op["name"]]
            if parts:
                row["Characteristics[OrganismPart]"] = parts[0]
                filled_by_col["Characteristics[OrganismPart]"] += 1
        
        # Disease
        if anchor["diseases"]:
            diseases = [d["name"].lower() for d in anchor["diseases"] if d["name"]]
            if diseases:
                row["Characteristics[Disease]"] = diseases[0]
                filled_by_col["Characteristics[Disease]"] += 1
        
        # Modifications (MOD → UNIMOD変換)
        mod_sdrfs = []
        has_iaa = False
        for mod in anchor["modifications"]:
            acc = mod.get("accession", "")
            if acc in MOD_TO_UNIMOD:
                mod_sdrfs.append(MOD_TO_UNIMOD[acc]["sdrf"])
            if acc in ALKYLATION_FROM_MOD:
                has_iaa = True
        
        # Modification columns
        mod_cols = ["Characteristics[Modification]"] + [f"Characteristics[Modification].{i}" for i in range(1, 7)]
        for i, sdrf_val in enumerate(mod_sdrfs):
            if i < len(mod_cols):
                row[mod_cols[i]] = sdrf_val
                filled_by_col[mod_cols[i]] += 1
        
        # AlkylationReagent from IAA
        if has_iaa:
            row["Characteristics[AlkylationReagent]"] = "iodoacetamide"
            filled_by_col["Characteristics[AlkylationReagent]"] += 1
        
        # Count fillable columns (exclude meta columns)
        meta_cols = {"", "ID", "PXD", "Raw Data File", "Usage"}
        for col in columns:
            if col in meta_cols:
                continue
            total_by_col[col] += 1
            total_cells += 1
            if row.get(col) and row[col] not in ("", "Not Applicable"):
                filled_cells += 1

# Report
print(f"=== アンカーのみで埋められたカラム ===")
print(f"Total fillable cells: {total_cells}")
print(f"Filled cells: {filled_cells}")
print(f"Fill rate: {filled_cells/total_cells*100:.1f}%")
print()

print(f"{'Column':<45} {'Filled':>6} / {'Total':>6} {'Rate':>7}")
print("-" * 70)
for col in columns:
    if col in {"", "ID", "PXD", "Raw Data File", "Usage"}:
        continue
    f = filled_by_col.get(col, 0)
    t = total_by_col.get(col, 0)
    rate = f/t*100 if t > 0 else 0
    if f > 0:
        print(f"{col:<45} {f:>6} / {t:>6} {rate:>6.1f}%")

print()
print(f"=== 埋まらなかったカラム ===")
unfilled = []
for col in columns:
    if col in {"", "ID", "PXD", "Raw Data File", "Usage"}:
        continue
    if filled_by_col.get(col, 0) == 0:
        unfilled.append(col)
print(f"{len(unfilled)} columns unfilled:")
for col in unfilled:
    print(f"  - {col}")

