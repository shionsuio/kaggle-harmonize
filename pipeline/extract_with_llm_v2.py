"""
Step 3 v2: 逐次累積型LLM抽出パイプライン。
前ステップの確定情報+推論候補を次ステップに引き継ぐ。

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python extract_with_llm_v2.py "Test PubText/Test PubText/"
"""
import json
import os
import sys
import glob
import re
import time

import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from chunk_pubtext import chunk_paper, get_llm_chunks

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# === ステップ定義 ===
STEPS = [
    {
        "name": "STEP1_OVERVIEW",
        "sections": ["TITLE", "ABSTRACT"],
        "description": "論文の概要から全体像を把握（生物種、疾患、細胞株、実験タイプ等）",
        "target_categories": [
            "Characteristics[Organism]",
            "Characteristics[OrganismPart]",
            "Characteristics[CellType]",
            "Characteristics[CellLine]",
            "Characteristics[Disease]",
            "Characteristics[MaterialType]",
            "Characteristics[Label]",
            "Characteristics[Specimen]",
            "Characteristics[Strain]",
            "Characteristics[GeneticModification]",
            "Characteristics[Treatment]",
            "Characteristics[DevelopmentalStage]",
            "Characteristics[Compound]",
            "Characteristics[ConcentrationOfCompound]",
            "Comment[AcquisitionMethod]",
        ],
        "extra_instructions": """
- CellLine: 不死化細胞株名（HEK293T, HeLa, C2C12, A549, U2OS等）を見逃さないこと。
  CellLineとCellTypeは異なる: CellLineは培養細胞株名、CellTypeは細胞の種類（neuron, fibroblast等）。
- Disease: "cancer", "tumor", "carcinoma" 等の疾患名に注意。
- Label: TMT, SILAC, iTRAQ, label-free 等の標識法。
""",
    },
    {
        "name": "STEP2_SAMPLE_PREP",
        "sections": ["METHODS"],
        "description": "サンプル調製・消化・修飾の詳細",
        "target_categories": [
            "Characteristics[CleavageAgent]",
            "Characteristics[AlkylationReagent]",
            "Characteristics[ReductionReagent]",
            "Characteristics[Label]",
            "Characteristics[Modification]",
            "Characteristics[Modification].1",
            "Characteristics[Modification].2",
            "Characteristics[Modification].3",
            "Characteristics[Modification].4",
            "Characteristics[Modification].5",
            "Characteristics[Modification].6",
            "Characteristics[Depletion]",
            "Characteristics[SpikedCompound]",
            "Characteristics[SyntheticPeptide]",
            "Comment[EnrichmentMethod]",
            "Comment[FractionationMethod]",
            "Comment[NumberOfFractions]",
            "Comment[Separation]",
        ],
    },
    {
        "name": "STEP3_MS_SETTINGS",
        "sections": ["METHODS"],
        "description": "質量分析装置・設定パラメータ",
        "target_categories": [
            "Comment[Instrument]",
            "Comment[FragmentationMethod]",
            "Comment[AcquisitionMethod]",
            "Comment[IonizationType]",
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
        "description": "rawファイル名とこれまでの知識を統合して、ファイルごとの割り当てを推定",
        "target_categories": [
            "Comment[FractionIdentifier]",
            "Characteristics[BiologicalReplicate]",
            "Characteristics[NumberOfBiologicalReplicates]",
            "Characteristics[NumberOfTechnicalReplicates]",
            "Characteristics[NumberOfSamples]",
            "FactorValue[Treatment]",
            "FactorValue[Temperature]",
            "FactorValue[Disease]",
            "FactorValue[GeneticModification]",
            "FactorValue[Compound]",
            "FactorValue[Bait]",
            "FactorValue[CellPart]",
            "FactorValue[ConcentrationOfCompound].1",
            "FactorValue[FractionIdentifier]",
        ],
        "extra_instructions": """
CRITICAL RULE FOR RAW FILE NAMES:
- ファイル名に明確なトークン（rep1, BR2, F3, DMSO, KO, 80C等）がある場合のみ推定する。
- 意味不明な連番（QExHF04026, 92-149139488等）や略語（ad_pl01等）からは推測しない。
- 読み取れないファイル名は正直に「Not Applicable」とする。間違った推測より無回答が安全。
- ファイルごとに異なる値がある場合は、per_file形式で出力する:
  "Comment[FractionIdentifier]": {
    "value": "per_file",
    "per_file": {"file1.raw": "1", "file2.raw": "2"},
    "evidence": "..."
  }
""",
    },
]


def build_anchor_context(anchor):
    """アンカー情報をテキストに整形"""
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
    if anchor.get("quantification_methods"):
        lines.append(f"Quantification: {', '.join(anchor['quantification_methods'])}")
    return "\n".join(lines)


def build_step_prompt(step, chunks, anchor_context, raw_files, accumulated_summary):
    """各ステップのプロンプトを構築"""

    # 該当セクションのテキスト収集
    section_texts = []
    for chunk in chunks:
        if chunk["section"] in step["sections"]:
            section_texts.append(f"[{chunk['section']}]\n{chunk['content']}")
    if not section_texts:
        return None

    combined_text = "\n\n".join(section_texts)
    categories_list = "\n".join(f"  - {c}" for c in step["target_categories"])

    raw_files_text = ""
    if raw_files and "Raw Data Files" in step["sections"]:
        raw_files_text = "\n[RAW DATA FILES]\n" + "\n".join(raw_files)

    # 累積サマリー（前ステップの結果）
    prev_context = ""
    if accumulated_summary:
        prev_context = f"""
[PREVIOUS STEPS - ACCUMULATED KNOWLEDGE]
{accumulated_summary}
"""

    extra = step.get("extra_instructions", "")
    extra_block = f"\n[STEP-SPECIFIC INSTRUCTIONS]\n{extra}" if extra else ""

    prompt = f"""あなたはプロテオミクスのSDRFメタデータ抽出の専門家です。

{anchor_context}
{prev_context}
[CURRENT TASK: {step['description']}]

以下のカテゴリについて、論文テキストから情報を抽出してください:
{categories_list}
{extra_block}

[論文テキスト]
{combined_text}
{raw_files_text}

[出力形式]
以下の3セクションに分けてJSON形式で出力してください:

{{
  "confirmed": {{
    "カテゴリ名": {{
      "value": "確定した値",
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
1. confirmedには、テキストまたはアンカーから確実に判断できるものだけを入れる。
2. hypothesesには、可能性はあるが確定できないものを入れる。前ステップの推論候補で、今回のテキストで確認できたものはconfirmedに昇格させる。
3. 前ステップで確定した情報と矛盾する場合は、矛盾をnoteフィールドに記録する。
4. rawファイル名のトークン分析では、ファイル名のどの部分からどう判断したか明記する。
5. summaryは次のステップが読むので、確定事項と未解決の推論を簡潔にまとめる。
6. 情報が論文テキストに存在しない場合、推測せずnot_applicableに入れる。幻覚は厳禁。
7. JSONのみ出力。説明文やmarkdownは不要。"""

    return prompt


def format_accumulated_summary(all_step_results):
    """これまでの全ステップの結果を累積サマリーに整形"""
    lines = []

    for step_name, result in all_step_results.items():
        if not isinstance(result, dict):
            continue

        lines.append(f"--- {step_name} ---")

        # 確定情報
        confirmed = result.get("confirmed", {})
        if confirmed:
            lines.append("Confirmed:")
            for cat, info in confirmed.items():
                if isinstance(info, dict):
                    val = info.get("value", "")
                    lines.append(f"  {cat}: {val}")
                else:
                    lines.append(f"  {cat}: {info}")

        # 推論候補
        hypotheses = result.get("hypotheses", {})
        if hypotheses:
            lines.append("Hypotheses (unresolved):")
            for cat, info in hypotheses.items():
                if isinstance(info, dict):
                    candidates = info.get("candidates", [])
                    reasoning = info.get("reasoning", "")
                    lines.append(f"  {cat}: candidates={candidates}, reason={reasoning}")
                else:
                    lines.append(f"  {cat}: {info}")

        # サマリー
        summary = result.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")

    return "\n".join(lines)


def call_claude(client, prompt, max_retries=5):
    """Claude API呼び出し（レートリミット対応リトライ付き）"""
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
                print(f"\n    Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_retries})...", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded")


def extract_paper_v2(client, pxd_id, pubtext_path, anchor_path, output_dir):
    """1論文を逐次累積型で処理"""
    print(f"\n{'='*60}")
    print(f"Processing {pxd_id} (v2 sequential)...")

    # チャンク分割
    paper = chunk_paper(pubtext_path)
    llm_chunks = paper["chunks"]  # 全チャンク（is_primary問わず）

    # アンカー読み込み
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

        # 累積サマリーを構築
        accumulated_summary = format_accumulated_summary(accumulated_results)

        # プロンプト構築
        prompt = build_step_prompt(
            step, llm_chunks, anchor_context,
            paper["raw_files"], accumulated_summary
        )

        if prompt is None:
            print("SKIP (no content)")
            continue

        try:
            response_text = call_claude(client, prompt)

            # JSON抽出
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                step_result = json.loads(json_match.group())
                results["steps"][step_name] = step_result
                accumulated_results[step_name] = step_result

                # ステータス表示
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

    # 保存
    out_file = os.path.join(output_dir, f"{pxd_id}_extraction_v2.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_file}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_llm_v2.py <pubtext_dir> [--dry-run]")
        sys.exit(1)

    pubtext_dir = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = None if dry_run else anthropic.Anthropic(api_key=api_key)

    output_dir = os.path.join(os.path.dirname(__file__), "llm_extractions")
    os.makedirs(output_dir, exist_ok=True)

    anchor_dir = os.path.join(os.path.dirname(__file__), "pride_anchors")

    txt_files = sorted(glob.glob(os.path.join(pubtext_dir, "*_PubText.txt")))
    if not txt_files:
        print(f"No files found in {pubtext_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} papers")
    print(f"Model: {MODEL}")
    print(f"Mode: {'dry-run' if dry_run else 'LIVE'}")
    print(f"Pipeline: Sequential accumulative (v2)")

    all_results = []
    for txt_file in txt_files:
        pxd = re.match(r"(PXD\d+)", os.path.basename(txt_file)).group(1)
        anchor_file = os.path.join(anchor_dir, f"{pxd}_anchor.json")

        if not os.path.exists(anchor_file):
            print(f"SKIP {pxd}: no anchor")
            continue

        result = extract_paper_v2(client, pxd, txt_file, anchor_file, output_dir)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_results)} papers processed.")


if __name__ == "__main__":
    main()
