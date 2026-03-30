"""
ローカルCV: v2抽出済みの訓練データ全てで評価
Usage: python run_cv.py
"""
import pandas as pd, json, sys, re, os, glob
sys.path.insert(0, os.path.dirname(__file__))

from score_function import load_sdrf, Harmonize_and_Evaluate_datasets
from build_submission_v2 import (
    get_value_from_v2, get_anchor_value, build_gt_vocab,
    fuzzy_snap, clean_value, DEFAULT_MODS, TMT_MODS, NEVER_GLOBAL,
    convert_extraction_v2, rule_based_supplement
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def run_cv():
    col_vocab, global_modes, non_na_ratio = build_gt_vocab()

    ext_dir = os.path.join(BASE_DIR, "pipeline", "llm_extractions")
    anchor_dir = os.path.join(BASE_DIR, "pipeline", "pride_anchors")
    gt_dir = os.path.join(BASE_DIR, "Training_SDRFs", "HarmonizedFiles")

    # v2抽出済み + 正解データがあるPXDを探す
    test_pxds = []
    for f in sorted(glob.glob(os.path.join(ext_dir, "*_extraction_v2.json"))):
        pxd = os.path.basename(f).replace("_extraction_v2.json", "")
        gt_path = os.path.join(gt_dir, f"Harmonized_{pxd}.csv")
        anchor_path = os.path.join(anchor_dir, f"{pxd}_anchor.json")
        if os.path.exists(gt_path) and os.path.exists(anchor_path):
            # エラー抽出（全ステップ失敗）を除外
            with open(f) as fh:
                ext = json.load(fh)
            has_data = any(
                isinstance(ext.get("steps", {}).get(s, {}), dict) and
                ext["steps"][s].get("confirmed")
                for s in ["STEP1_OVERVIEW", "STEP2_SAMPLE_PREP", "STEP3_MS_SETTINGS"]
            )
            if has_data:
                test_pxds.append(pxd)

    print(f"CV on {len(test_pxds)} papers: {test_pxds}")
    all_f1s = []

    for pxd in test_pxds:
        sol = pd.read_csv(os.path.join(gt_dir, f"Harmonized_{pxd}.csv"))
        if "ID" in sol.columns:
            sol = sol.drop(columns=["ID"])
        with open(os.path.join(ext_dir, f"{pxd}_extraction_v2.json")) as f:
            ext = convert_extraction_v2(json.load(f))
        with open(os.path.join(anchor_dir, f"{pxd}_anchor.json")) as f:
            anchor = json.load(f)

        # 文脈情報
        is_tmt = False
        has_trypsin = False
        has_enzyme_digest = True
        has_fractionation = False
        has_alkylation = False

        lv = get_value_from_v2(ext, "Characteristics[Label]")
        if lv and "TMT" in str(lv).upper(): is_tmt = True
        cv = get_value_from_v2(ext, "Characteristics[CleavageAgent]")
        if cv:
            cl = str(cv).lower()
            if "trypsin" in cl or "lys-c" in cl: has_trypsin = True
            if "unspecific" in cl: has_enzyme_digest = False
        av = get_value_from_v2(ext, "Characteristics[AlkylationReagent]")
        if av and av != "Not Applicable": has_alkylation = True
        for step_data in ext.get("steps", {}).values():
            if not isinstance(step_data, dict): continue
            for entry in step_data.get("confirmed", {}).values():
                if isinstance(entry, dict):
                    ev = str(entry.get("evidence", "")).lower()
                    if "iodoacetamide" in ev or "chloroacetamide" in ev or "alkylat" in ev:
                        has_alkylation = True
        fv = get_value_from_v2(ext, "Comment[FractionationMethod]")
        if fv and fv != "Not Applicable":
            fl = str(fv).lower()
            if "no frac" not in fl and "not" not in fl: has_fractionation = True

        # ルールベース補完
        pubtext_path = os.path.join(BASE_DIR, "Training_PubText", "PubText", f"{pxd}_PubText.txt")
        if not os.path.exists(pubtext_path):
            pubtext_path = os.path.join(BASE_DIR, "Test PubText", "Test PubText", f"{pxd}_PubText.txt")
        rule_supplements = rule_based_supplement(ext, pubtext_path) if os.path.exists(pubtext_path) else {}

        sub_rows = []
        for _, gt_row in sol.iterrows():
            raw_file = gt_row.get("Raw Data File", "")
            sub_row = {"PXD": pxd}
            for col in sol.columns:
                if col in ("PXD", "Raw Data File", "Usage", ""): continue
                base_col = re.sub(r'\.\d+$', '', col)
                value = None

                # アンカー優先
                if col in ("Characteristics[Organism]", "Comment[Instrument]"):
                    value = get_anchor_value(anchor, col)
                if not value:
                    value = get_value_from_v2(ext, col, raw_file)
                if not value or value == "Not Applicable":
                    a = get_anchor_value(anchor, col)
                    if a: value = a
                # ルールベース補完（LLMがNot Applicableでも上書き）
                if (not value or value == "Not Applicable") and col in rule_supplements:
                    value = rule_supplements[col]
                # 条件付きデフォルト
                if not value:
                    if col == "Characteristics[Modification]" and (has_trypsin or has_alkylation):
                        value = DEFAULT_MODS[col]
                    elif col == "Characteristics[Modification].1" and has_enzyme_digest:
                        value = DEFAULT_MODS[col]
                    elif col == "Characteristics[Modification].2" and is_tmt:
                        value = TMT_MODS[col]
                    elif col == "Characteristics[Modification].3" and is_tmt:
                        value = TMT_MODS[col]
                    elif col == "Comment[FractionIdentifier]" and not has_fractionation:
                        value = "1"
                    elif col == "Characteristics[BiologicalReplicate]":
                        value = "1"
                    elif base_col not in NEVER_GLOBAL and non_na_ratio.get(col, 0) > 0.80:
                        value = global_modes.get(col, "Not Applicable")
                if not value: value = "Not Applicable"
                value = clean_value(value, col)
                if value == "Not Applicable":
                    if col == "Characteristics[Modification]" and (has_trypsin or has_alkylation):
                        value = DEFAULT_MODS[col]
                    elif col == "Characteristics[Modification].1" and has_enzyme_digest:
                        value = DEFAULT_MODS[col]
                    elif col == "Comment[FractionIdentifier]" and not has_fractionation:
                        value = "1"
                    elif col == "Characteristics[BiologicalReplicate]":
                        value = "1"
                if value != "Not Applicable":
                    value = fuzzy_snap(value, base_col, col_vocab)
                if col == "Characteristics[Modification]" and (has_trypsin or has_alkylation):
                    if "carbamidomethyl" not in value.lower():
                        value = DEFAULT_MODS[col]
                sub_row[col] = value
            sub_rows.append(sub_row)

        sub_df = pd.DataFrame(sub_rows)
        sol_dict = load_sdrf(sol)
        sub_dict = load_sdrf(sub_df)
        _, _, eval_df = Harmonize_and_Evaluate_datasets(sol_dict, sub_dict, threshold=0.80)
        f1 = eval_df['f1'].dropna().mean()
        all_f1s.append(f1)

        print(f"\n{pxd}: F1={f1:.4f} (TMT={is_tmt}, trypsin={has_trypsin})")
        for _, row in eval_df.sort_values("f1").iterrows():
            if row['f1'] < 1.0:
                print(f"  {row['AnnotationType']:<50} F1={row['f1']:.3f}")

    print(f"\n{'='*60}")
    print(f"Average CV F1: {sum(all_f1s)/len(all_f1s):.4f} ({len(all_f1s)} papers)")
    for pxd, f1 in zip(test_pxds, all_f1s):
        print(f"  {pxd}: {f1:.4f}")

if __name__ == "__main__":
    run_cv()
