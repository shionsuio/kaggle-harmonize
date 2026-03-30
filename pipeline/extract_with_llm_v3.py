"""
Step 3 v3: 逐次累積型LLM抽出パイプライン（GT候補リスト付き）
候補リストから選ばせることでフォーマット不一致を解消する。

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python extract_with_llm_v3.py "Test PubText/Test PubText/"
"""
import json
import os
import sys
import glob
import re
import time

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from chunk_pubtext import chunk_paper

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# === GT候補リスト読み込み ===
GT_CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "gt_candidates.json")
with open(GT_CANDIDATES_PATH) as f:
    GT_CANDIDATES = json.load(f)

# === ステップ定義 ===
STEPS = [
    {
        "name": "STEP1_OVERVIEW",
        "sections": ["TITLE", "ABSTRACT"],
        "description": "論文の概要から生物学的コンテキストを抽出",
        "categories": [
            "Characteristics[Organism]",
            "Characteristics[OrganismPart]",
            "Characteristics[CellType]",
            "Characteristics[CellLine]",
            "Characteristics[Disease]",
            "Characteristics[MaterialType]",
            "Characteristics[Label]",
            "Characteristics[DevelopmentalStage]",
            "Characteristics[Specimen]",
            "Characteristics[Strain]",
            "Characteristics[Sex]",
            "Characteristics[Compound]",
            "Comment[AcquisitionMethod]",
        ],
    },
    {
        "name": "STEP2_SAMPLE_PREP",
        "sections": ["METHODS"],
        "description": "サンプル調製・消化・修飾の詳細",
        "categories": [
            "Characteristics[CleavageAgent]",
            "Characteristics[AlkylationReagent]",
            "Characteristics[ReductionReagent]",
            "Characteristics[Label]",
            "Characteristics[Modification]",
            "Characteristics[Depletion]",
            "Comment[EnrichmentMethod]",
            "Comment[FractionationMethod]",
            "Comment[Separation]",
        ],
    },
    {
        "name": "STEP3_MS_SETTINGS",
        "sections": ["METHODS"],
        "description": "質量分析装置・設定パラメータ",
        "categories": [
            "Comment[Instrument]",
            "Comment[FragmentationMethod]",
            "Comment[AcquisitionMethod]",
            "Comment[MS2MassAnalyzer]",
            "Comment[CollisionEnergy]",
            "Comment[PrecursorMassTolerance]",
            "Comment[FragmentMassTolerance]",
            "Comment[NumberOfMissedCleavages]",
            "Comment[GradientTime]",
            "Comment[FlowRateChromatogram]",
        ],
    },
    {
        "name": "STEP4_RAW_FILES",
        "sections": ["Raw Data Files", "METHODS"],
        "description": "rawファイル名から実験条件を推定",
        "categories": [
            "Comment[FractionIdentifier]",
            "Characteristics[BiologicalReplicate]",
            "FactorValue[Treatment]",
            "FactorValue[Temperature]",
            "FactorValue[Disease]",
            "FactorValue[GeneticModification]",
            "FactorValue[Compound]",
            "FactorValue[Bait]",
        ],
    },
]


def build_anchor_context(anchor):
    lines = ["[CONFIRMED FROM PRIDE DATABASE]"]
    if anchor.get("organism"):
        lines.append(f"Organism: {', '.join(o['name'] for o in anchor['organism'])}")
    if anchor.get("instruments"):
        for inst in anchor["instruments"]:
            lines.append(f"Instrument: {inst['name']} ({inst['accession']})")
    if anchor.get("inferred_fragmentation"):
        lines.append(f"Fragmentation (from instrument): {anchor['inferred_fragmentation']}")
    if anchor.get("inferred_ms2_analyzer"):
        lines.append(f"MS2 Analyzer (from instrument): {anchor['inferred_ms2_analyzer']}")
    if anchor.get("inferred_ionization"):
        lines.append(f"Ionization (from instrument): {anchor['inferred_ionization']}")
    if anchor.get("organism_parts"):
        lines.append(f"Organism Part: {', '.join(op['name'] for op in anchor['organism_parts'])}")
    if anchor.get("diseases"):
        lines.append(f"Disease: {', '.join(d['name'] for d in anchor['diseases'])}")
    if anchor.get("modifications"):
        for mod in anchor["modifications"]:
            lines.append(f"Modification: {mod['name']} ({mod['accession']})")
    if anchor.get("experiment_types"):
        lines.append(f"Experiment Type: {', '.join(anchor['experiment_types'])}")
    return "\n".join(lines)


def build_candidates_block(categories):
    """対象カテゴリのGT候補リストをプロンプト用テキストに整形"""
    lines = ["[ALLOWED VALUES — 以下の候補から選んでください。候補にない値は使わないでください]"]
    for cat in categories:
        base = re.sub(r'\.\d+$', '', cat)
        if base in GT_CANDIDATES:
            cands = GT_CANDIDATES[base][:30]  # 最大30個
            if cands:
                lines.append(f"\n{cat}:")
                lines.append(f"  {', '.join(cands[:15])}")
                if len(cands) > 15:
                    lines.append(f"  {', '.join(cands[15:30])}")
    return "\n".join(lines)


def build_step_prompt(step, chunks, anchor_context, raw_files, accumulated_summary):
    section_texts = []
    for chunk in chunks:
        if chunk["section"] in step["sections"]:
            section_texts.append(f"[{chunk['section']}]\n{chunk['content']}")
    if not section_texts:
        return None

    combined_text = "\n\n".join(section_texts)
    categories = step["categories"]
    categories_list = "\n".join(f"  - {c}" for c in categories)

    raw_files_text = ""
    if raw_files and "Raw Data Files" in step["sections"]:
        raw_files_text = "\n[RAW DATA FILES]\n" + "\n".join(raw_files)

    prev_context = ""
    if accumulated_summary:
        prev_context = f"\n[PREVIOUS STEPS - ACCUMULATED KNOWLEDGE]\n{accumulated_summary}\n"

    candidates_block = build_candidates_block(categories)

    prompt = f"""あなたはプロテオミクスのSDRFメタデータ抽出の専門家です。

{anchor_context}
{prev_context}
[CURRENT TASK: {step['description']}]

以下のカテゴリについて、論文テキストから情報を抽出してください:
{categories_list}

{candidates_block}

[論文テキスト]
{combined_text}
{raw_files_text}

[出力形式]
JSON形式で出力してください:

{{
  "confirmed": {{
    "カテゴリ名": {{
      "value": "候補リストから選んだ値",
      "evidence": "根拠となるテキスト引用",
      "source": "manuscript/anchor/filename"
    }}
  }},
  "hypotheses": {{
    "カテゴリ名": {{
      "candidates": ["候補1", "候補2"],
      "reasoning": "なぜこれらが候補なのか",
      "needs": "確定に必要な追加情報"
    }}
  }},
  "not_applicable": ["該当しないカテゴリ名のリスト"],
  "summary": "このステップで分かったことの要約（次ステップへの引き継ぎ用、3-5文）"
}}

[ルール]
1. valueには必ず候補リスト（ALLOWED VALUES）に記載された値を使う。候補にない値は使わない。
2. 候補リストにぴったり合う値がなければ、最も近い候補を選ぶ。
3. Modificationは候補のNT名（例: Carbamidomethyl, Oxidation）を使う。複数あれば別エントリで出力。
4. confirmedには確実に判断できるものだけ。不確実なものはhypotheses。
5. 前ステップの推論候補で今回確認できたものはconfirmedに昇格。
6. 情報が論文テキストに存在しない場合はnot_applicableに入れる。推測しない。
7. rawファイル名の意味不明な連番からは推測しない。明確なトークン（rep, BR, DMSO等）のみ使う。
8. JSONのみ出力。説明文やmarkdownは不要。"""

    return prompt


def format_accumulated_summary(all_step_results):
    lines = []
    for step_name, result in all_step_results.items():
        if not isinstance(result, dict):
            continue
        lines.append(f"--- {step_name} ---")
        confirmed = result.get("confirmed", {})
        if confirmed:
            lines.append("Confirmed:")
            for cat, info in confirmed.items():
                if isinstance(info, dict):
                    val = info.get("value", "")
                    lines.append(f"  {cat}: {val}")
                else:
                    lines.append(f"  {cat}: {info}")
        hypotheses = result.get("hypotheses", {})
        if hypotheses:
            lines.append("Hypotheses:")
            for cat, info in hypotheses.items():
                if isinstance(info, dict):
                    candidates = info.get("candidates", [])
                    lines.append(f"  {cat}: candidates={candidates}")
                else:
                    lines.append(f"  {cat}: {info}")
        summary = result.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")
    return "\n".join(lines)


def call_claude(client, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"\n    Rate limit, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")


def extract_paper_v3(client, pxd_id, pubtext_path, anchor_path, output_dir):
    print(f"\n{'='*60}")
    print(f"Processing {pxd_id} (v3 with GT candidates)...")

    paper = chunk_paper(pubtext_path)
    llm_chunks = paper["chunks"]

    with open(anchor_path) as f:
        anchor = json.load(f)
    anchor_context = build_anchor_context(anchor)

    results = {
        "pxd": pxd_id,
        "raw_files": paper["raw_files"],
        "anchor": anchor,
        "steps": {},
    }

    accumulated_results = {}

    for step in STEPS:
        step_name = step["name"]
        print(f"  [{step_name}] {step['description']}...", end=" ")

        accumulated_summary = format_accumulated_summary(accumulated_results)

        prompt = build_step_prompt(
            step, llm_chunks, anchor_context,
            paper["raw_files"], accumulated_summary
        )

        if prompt is None:
            print("SKIP (no content)")
            continue

        try:
            response_text = call_claude(client, prompt)

            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                step_result = json.loads(json_match.group())
                results["steps"][step_name] = step_result
                accumulated_results[step_name] = step_result

                n_confirmed = len(step_result.get("confirmed", {}))
                n_hypo = len(step_result.get("hypotheses", {}))
                n_na = len(step_result.get("not_applicable", []))
                print(f"OK (confirmed={n_confirmed}, hypotheses={n_hypo}, N/A={n_na})")
            else:
                print("WARN: no JSON")
                results["steps"][step_name] = {"_raw": response_text}

        except json.JSONDecodeError as e:
            print(f"JSON error: {e}")
            results["steps"][step_name] = {"_raw": response_text}
        except Exception as e:
            print(f"ERROR: {e}")
            results["steps"][step_name] = {"_error": str(e)}

        time.sleep(1)

    out_file = os.path.join(output_dir, f"{pxd_id}_extraction_v3.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_file}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_llm_v3.py <pubtext_dir>")
        sys.exit(1)

    pubtext_dir = sys.argv[1]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    output_dir = os.path.join(os.path.dirname(__file__), "llm_extractions")
    os.makedirs(output_dir, exist_ok=True)
    anchor_dir = os.path.join(os.path.dirname(__file__), "pride_anchors")

    txt_files = sorted(glob.glob(os.path.join(pubtext_dir, "*_PubText.txt")))
    if not txt_files:
        print(f"No files found in {pubtext_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} papers")
    print(f"Model: {MODEL}")
    print(f"Pipeline: v3 (GT candidate lists)")

    for txt_file in txt_files:
        pxd = re.match(r"(PXD\d+)", os.path.basename(txt_file)).group(1)
        anchor_file = os.path.join(anchor_dir, f"{pxd}_anchor.json")

        if not os.path.exists(anchor_file):
            print(f"SKIP {pxd}: no anchor")
            continue

        extract_paper_v3(client, pxd, txt_file, anchor_file, output_dir)


if __name__ == "__main__":
    main()
