"""
Step 3: アンカー + チャンクをClaude APIに渡してSDRFメタデータを抽出する。
チャンクごとにカテゴリグループを絞って投げる。

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python extract_with_llm.py "Test PubText/Test PubText/"
"""
import json
import os
import sys
import glob
import re
import time

import anthropic

from chunk_pubtext import chunk_paper, get_llm_chunks

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# --- カテゴリグループ定義 ---
# 各チャンク(セクション)に対して抽出すべきカテゴリを定義

CATEGORY_GROUPS = {
    "TITLE_ABSTRACT": {
        "sections": ["TITLE", "ABSTRACT"],
        "categories": [
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
            "Comment[AcquisitionMethod]",
        ],
        "description": "論文の概要から生物学的コンテキストを抽出",
    },
    "METHODS_SAMPLE": {
        "sections": ["METHODS"],
        "categories": [
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
        "description": "サンプル調製・消化・修飾に関する情報を抽出",
    },
    "METHODS_MS": {
        "sections": ["METHODS"],
        "categories": [
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
        "description": "質量分析装置・設定パラメータを抽出",
    },
    "RAW_FILES": {
        "sections": ["Raw Data Files"],
        "categories": [
            "Comment[FractionIdentifier]",
            "Characteristics[BiologicalReplicate]",
            "Characteristics[NumberOfBiologicalReplicates]",
            "Characteristics[NumberOfTechnicalReplicates]",
            "Characteristics[NumberOfSamples]",
        ],
        "description": "rawファイル名からフラクション・レプリケート情報を推定",
    },
}


def build_anchor_context(anchor):
    """アンカー情報をLLMに渡すテキストに整形"""
    lines = ["[CONFIRMED METADATA FROM PRIDE DATABASE - Use as ground truth]"]

    if anchor.get("organism"):
        orgs = ", ".join(o["name"] for o in anchor["organism"])
        lines.append(f"Organism: {orgs}")

    if anchor.get("instruments"):
        for inst in anchor["instruments"]:
            lines.append(f"Instrument: {inst['name']} ({inst['accession']})")

    if anchor.get("inferred_fragmentation"):
        lines.append(f"Fragmentation (inferred from instrument): {anchor['inferred_fragmentation']}")

    if anchor.get("inferred_ms2_analyzer"):
        lines.append(f"MS2 Analyzer (inferred from instrument): {anchor['inferred_ms2_analyzer']}")

    if anchor.get("inferred_ionization"):
        lines.append(f"Ionization (inferred from instrument): {anchor['inferred_ionization']}")

    if anchor.get("organism_parts"):
        parts = ", ".join(op["name"] for op in anchor["organism_parts"])
        lines.append(f"Organism Part: {parts}")

    if anchor.get("diseases"):
        diseases = ", ".join(d["name"] for d in anchor["diseases"])
        lines.append(f"Disease: {diseases}")

    if anchor.get("modifications"):
        for mod in anchor["modifications"]:
            lines.append(f"Modification (PRIDE): {mod['name']} ({mod['accession']})")

    if anchor.get("experiment_types"):
        lines.append(f"Experiment Type: {', '.join(anchor['experiment_types'])}")

    if anchor.get("quantification_methods"):
        lines.append(f"Quantification: {', '.join(anchor['quantification_methods'])}")

    if anchor.get("raw_files"):
        lines.append(f"Raw files in dataset: {len(anchor['raw_files'])} files")

    return "\n".join(lines)


def build_extraction_prompt(group_name, group_config, chunks, anchor_context, raw_files):
    """カテゴリグループ用のプロンプトを構築"""

    # 該当セクションのテキストを収集
    section_texts = []
    for chunk in chunks:
        if chunk["section"] in group_config["sections"]:
            section_texts.append(f"[{chunk['section']}]\n{chunk['content']}")

    if not section_texts:
        return None

    combined_text = "\n\n".join(section_texts)
    categories_list = "\n".join(f"  - {c}" for c in group_config["categories"])

    # rawファイルリスト（RAW_FILESグループの場合は特に重要）
    raw_files_text = ""
    if raw_files:
        raw_files_text = f"\n[RAW DATA FILES]\n" + "\n".join(raw_files)

    prompt = f"""あなたはプロテオミクスのSDRFメタデータ抽出の専門家です。
論文テキストから以下のカテゴリの値を抽出してください。

{anchor_context}

[抽出対象カテゴリ]
{categories_list}

[論文テキスト]
{combined_text}
{raw_files_text}

[出力ルール]
1. JSON形式で出力。キーはカテゴリ名、値は抽出したテキスト。
2. 各カテゴリについて:
   - "value": 抽出した値
   - "evidence": 論文テキスト中の根拠となる箇所（引用）
   - "confidence": 確信度 (high/medium/low)
   - "source": 情報源 (manuscript/filename/anchor/inferred)
3. 論文テキストに情報がない場合は "value": "Not Applicable" とする。
4. アンカー情報と矛盾する値を抽出した場合は、アンカー情報を優先しつつ、矛盾を "note" フィールドに記録する。
5. Modificationは論文中の記述そのままで良い（UNIMOD変換は後処理で行う）。
6. rawファイル名からフラクション/レプリケート情報を推定する場合、ファイル名のどのトークンから判断したか明記する。
7. 全rawファイルに共通する値と、ファイルごとに異なる値を区別する。
   - "applies_to": "all" (全ファイル共通) or {{"file_pattern": "説明"}} (ファイルごと)

出力はJSONのみ。説明文やmarkdownは不要。"""

    return prompt


def call_claude(client, prompt):
    """Claude APIを呼び出す"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def extract_paper(client, pxd_id, pubtext_path, anchor_path, output_dir):
    """1論文を処理"""
    print(f"\n{'='*60}")
    print(f"Processing {pxd_id}...")

    # チャンク分割
    paper = chunk_paper(pubtext_path)
    llm_chunks = get_llm_chunks(paper)

    # アンカー読み込み
    with open(anchor_path) as f:
        anchor = json.load(f)
    anchor_context = build_anchor_context(anchor)

    results = {
        "pxd": pxd_id,
        "raw_files": paper["raw_files"],
        "anchor": anchor,
        "extractions": {},
    }

    # カテゴリグループごとにLLM呼び出し
    for group_name, group_config in CATEGORY_GROUPS.items():
        print(f"  [{group_name}] {group_config['description']}...", end=" ")

        prompt = build_extraction_prompt(
            group_name, group_config, llm_chunks, anchor_context, paper["raw_files"]
        )

        if prompt is None:
            print("SKIP (no content)")
            continue

        try:
            response_text = call_claude(client, prompt)

            # JSON抽出（レスポンスにmarkdownが混ざる場合の対策）
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                extraction = json.loads(json_match.group())
                results["extractions"][group_name] = extraction
                print(f"OK ({len(extraction)} categories)")
            else:
                print(f"WARN: no JSON found")
                results["extractions"][group_name] = {"_raw": response_text}

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            results["extractions"][group_name] = {"_raw": response_text}
        except Exception as e:
            print(f"ERROR: {e}")
            results["extractions"][group_name] = {"_error": str(e)}

        time.sleep(1)  # rate limit対策

    # 保存
    out_file = os.path.join(output_dir, f"{pxd_id}_extraction.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_file}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_with_llm.py <pubtext_dir> [--dry-run]")
        sys.exit(1)

    pubtext_dir = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    # API キー確認
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = None if dry_run else anthropic.Anthropic(api_key=api_key)

    # 出力ディレクトリ
    output_dir = os.path.join(os.path.dirname(__file__), "llm_extractions")
    os.makedirs(output_dir, exist_ok=True)

    # アンカーディレクトリ
    anchor_dir = os.path.join(os.path.dirname(__file__), "pride_anchors")

    # PubTextファイル一覧
    txt_files = sorted(glob.glob(os.path.join(pubtext_dir, "*_PubText.txt")))
    if not txt_files:
        print(f"No *_PubText.txt files found in {pubtext_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} papers")
    print(f"Model: {MODEL}")
    print(f"Dry run: {dry_run}")

    if dry_run:
        # プロンプトだけ表示
        for txt_file in txt_files[:1]:  # 1本だけ
            pxd = re.match(r"(PXD\d+)", os.path.basename(txt_file)).group(1)
            paper = chunk_paper(txt_file)
            llm_chunks = get_llm_chunks(paper)

            anchor_file = os.path.join(anchor_dir, f"{pxd}_anchor.json")
            with open(anchor_file) as f:
                anchor = json.load(f)
            anchor_context = build_anchor_context(anchor)

            for group_name, group_config in CATEGORY_GROUPS.items():
                prompt = build_extraction_prompt(
                    group_name, group_config, llm_chunks, anchor_context, paper["raw_files"]
                )
                if prompt:
                    print(f"\n{'='*60}")
                    print(f"[{pxd}] {group_name}")
                    print(f"{'='*60}")
                    print(prompt[:2000])
                    print(f"... ({len(prompt)} chars total)")
        return

    # 本番実行
    all_results = []
    for txt_file in txt_files:
        pxd = re.match(r"(PXD\d+)", os.path.basename(txt_file)).group(1)
        anchor_file = os.path.join(anchor_dir, f"{pxd}_anchor.json")

        if not os.path.exists(anchor_file):
            print(f"SKIP {pxd}: no anchor file")
            continue

        result = extract_paper(client, pxd, txt_file, anchor_file, output_dir)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_results)} papers processed.")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
