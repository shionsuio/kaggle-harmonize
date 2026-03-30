"""
Step 6: LLM抽出結果 + アンカー + フォーマット変換 → submission.csv 生成

Usage:
    python build_submission.py
"""
import json
import csv
import os
import re
import sys
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from format_converter import convert_extraction_v2


def load_sample_submission(path):
    """SampleSubmission.csvを読み込み"""
    with open(path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    return columns, rows


def load_extraction_v2(path):
    """v2抽出結果を読み込み"""
    with open(path) as f:
        return json.load(f)


def load_anchor(path):
    """アンカー情報を読み込み"""
    with open(path) as f:
        return json.load(f)


def _extract_entry_value(entry, raw_file=None):
    """entryから値を取得するヘルパー"""
    if isinstance(entry, dict):
        val = entry.get("value", "")
        if not val:
            candidates = entry.get("candidates", [])
            val = candidates[0] if candidates else ""
        if val == "per_file" and raw_file:
            per_file = entry.get("per_file", {})
            return per_file.get(raw_file, "Not Applicable")
        return val if val else "Not Applicable"
    return str(entry)


def get_value_from_v2(ext, category, raw_file=None):
    """
    v2形式から値を取得。
    全ステップを走査し、confirmed > hypothesis の優先度で最良の値を返す。
    後ステップのconfirmedは前ステップのhypothesisより優先される。
    """
    best_confirmed = None
    best_hypothesis = None
    is_na = False

    for step_name in ["STEP1_OVERVIEW", "STEP2_SAMPLE_PREP", "STEP3_MS_SETTINGS", "STEP4_RAW_FILES"]:
        step_data = ext.get("steps", {}).get(step_name, {})
        if not isinstance(step_data, dict):
            continue

        # confirmed（後ステップが優先 = 上書き）
        confirmed = step_data.get("confirmed", {})
        if category in confirmed:
            best_confirmed = _extract_entry_value(confirmed[category], raw_file)

        # hypotheses（confirmedがなければ使う、後ステップが優先）
        hypotheses = step_data.get("hypotheses", {})
        if category in hypotheses:
            best_hypothesis = _extract_entry_value(hypotheses[category], raw_file)

        # not_applicable
        na_list = step_data.get("not_applicable", [])
        if category in na_list:
            is_na = True

    # 優先度: confirmed > hypothesis > not_applicable
    if best_confirmed and best_confirmed != "Not Applicable":
        return best_confirmed
    if best_hypothesis and best_hypothesis != "Not Applicable":
        return best_hypothesis
    if is_na:
        return "Not Applicable"
    return None


def get_anchor_value(anchor, category):
    """アンカーから値を取得"""
    if category == "Characteristics[Organism]":
        orgs = anchor.get("organism", [])
        if orgs:
            name = orgs[0]["name"]
            # "(mouse)", "(human)" 等の括弧を除去
            name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
            return name.lower()
        return None

    if category == "Comment[Instrument]":
        insts = anchor.get("instruments", [])
        return insts[0].get("sdrf_format", "") if insts else None

    if category == "Comment[FragmentationMethod]":
        return anchor.get("inferred_fragmentation")

    if category == "Comment[MS2MassAnalyzer]":
        return anchor.get("inferred_ms2_analyzer")

    if category == "Comment[IonizationType]":
        return anchor.get("inferred_ionization")

    if category == "Characteristics[OrganismPart]":
        parts = anchor.get("organism_parts", [])
        return parts[0]["name"].lower() if parts else None

    if category == "Characteristics[Disease]":
        diseases = anchor.get("diseases", [])
        return diseases[0]["name"].lower() if diseases else None

    return None


def build_submission():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sample_sub_path = os.path.join(base_dir, "SampleSubmission.csv")
    extraction_dir = os.path.join(base_dir, "pipeline", "llm_extractions")
    anchor_dir = os.path.join(base_dir, "pipeline", "pride_anchors")
    output_path = os.path.join(base_dir, "submission.csv")

    # Load submission template
    columns, rows = load_sample_submission(sample_sub_path)

    # Group rows by PXD
    pxd_groups = defaultdict(list)
    for i, row in enumerate(rows):
        pxd_groups[row["PXD"]].append(i)

    meta_cols = {"", "ID", "PXD", "Raw Data File", "Usage"}
    # アンカーを優先すべきカラム（PRIDEの構造化データが最も信頼できる）
    ANCHOR_PRIORITY_COLS = {
        "Characteristics[Organism]",
        "Comment[Instrument]",
    }
    filled_count = 0
    total_count = 0

    for pxd, row_indices in pxd_groups.items():
        # Load extraction
        ext_path = os.path.join(extraction_dir, f"{pxd}_extraction_v2.json")
        anchor_path = os.path.join(anchor_dir, f"{pxd}_anchor.json")

        ext = None
        anchor = None

        if os.path.exists(ext_path):
            ext = load_extraction_v2(ext_path)
            ext = convert_extraction_v2(ext)

        if os.path.exists(anchor_path):
            anchor = load_anchor(anchor_path)

        if not ext and not anchor:
            print(f"  SKIP {pxd}: no extraction or anchor")
            continue

        print(f"  {pxd}: {len(row_indices)} rows...", end=" ")
        pxd_filled = 0

        for idx in row_indices:
            row = rows[idx]
            raw_file = row.get("Raw Data File", "")

            for col in columns:
                if col in meta_cols:
                    continue

                total_count += 1
                value = None

                # アンカー優先カラム: アンカーがあればそちらを使う
                if col in ANCHOR_PRIORITY_COLS and anchor:
                    anchor_val = get_anchor_value(anchor, col)
                    if anchor_val:
                        value = anchor_val

                # LLM抽出から取得（アンカー優先カラムでアンカーが取れなかった場合も含む）
                if value is None and ext:
                    value = get_value_from_v2(ext, col, raw_file)

                # アンカーからフォールバック（アンカー優先カラム以外）
                if (value is None or value == "Not Applicable") and anchor:
                    anchor_val = get_anchor_value(anchor, col)
                    if anchor_val:
                        value = anchor_val

                # デフォルト
                if value is None or value == "":
                    value = "Not Applicable"

                # TextSpan を置換
                if row[col] == "TextSpan" or row[col] == "":
                    row[col] = value
                    if value != "Not Applicable":
                        filled_count += 1
                        pxd_filled += 1

        print(f"{pxd_filled} cells filled")

    # Write submission
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n{'='*50}")
    print(f"Submission saved: {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Filled cells: {filled_count} / {total_count}")
    print(f"Fill rate: {filled_count/total_count*100:.1f}%")


if __name__ == "__main__":
    build_submission()
