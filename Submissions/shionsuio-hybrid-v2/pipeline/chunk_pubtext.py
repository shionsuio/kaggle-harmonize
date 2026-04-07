"""
Step 2: PubText.txt をセクション単位でチャンク分割する。
空セクションは除外し、LLMに渡すべきチャンクのみ返す。

Usage:
    from chunk_pubtext import chunk_paper
    chunks = chunk_paper("path/to/PXD000070_PubText.txt")
"""
import re
import os
import json
import glob
import sys

# セクションヘッダのパターン
SECTION_HEADERS = [
    "TITLE",
    "ABSTRACT",
    "INTRO",
    "RESULTS",
    "DISCUSS",
    "FIG",
    "METHODS",
    "Raw Data Files",
]

# LLMに渡すセクション（優先度順）
PRIMARY_SECTIONS = ["TITLE", "ABSTRACT", "METHODS", "Raw Data Files"]
# METHODSが空の場合のフォールバック
FALLBACK_SECTIONS = ["INTRO", "RESULTS"]

# 空判定の最小文字数
MIN_CONTENT_LENGTH = 20


def chunk_paper(txt_path):
    """
    PubText.txt をセクション分割し、有効なチャンクを返す。

    Returns:
        dict: {
            "pxd": "PXD000070",
            "file": "path/to/file.txt",
            "chunks": [
                {"section": "TITLE", "content": "...", "char_count": 100, "is_primary": True},
                ...
            ],
            "empty_sections": ["DISCUSS", "FIG"],
            "has_methods": True,
            "raw_files": ["file1.raw", "file2.raw"],
        }
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # PXD ID をファイル名から抽出
    basename = os.path.basename(txt_path)
    pxd_match = re.match(r"(PXD\d+)", basename)
    pxd_id = pxd_match.group(1) if pxd_match else "unknown"

    # セクション分割
    sections = {}
    current_section = None
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        # セクションヘッダ判定
        header_match = None
        for h in SECTION_HEADERS:
            if stripped == f"{h}:" or stripped == h:
                header_match = h
                break

        if header_match:
            # 前のセクションを保存
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = header_match
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    # 最後のセクションを保存
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    # チャンク構築
    chunks = []
    empty_sections = []
    raw_files = []

    for section_name in SECTION_HEADERS:
        content = sections.get(section_name, "").strip()

        # Raw Data Files は特別処理
        if section_name == "Raw Data Files":
            if content:
                raw_files = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip() and line.strip().endswith(".raw")
                ]
                chunks.append({
                    "section": section_name,
                    "content": content,
                    "char_count": len(content),
                    "is_primary": True,
                    "raw_file_count": len(raw_files),
                })
            else:
                empty_sections.append(section_name)
            continue

        # 空判定
        if len(content) < MIN_CONTENT_LENGTH:
            empty_sections.append(section_name)
            continue

        is_primary = section_name in PRIMARY_SECTIONS
        chunks.append({
            "section": section_name,
            "content": content,
            "char_count": len(content),
            "is_primary": is_primary,
        })

    # METHODSが空の場合、フォールバックセクションをprimaryに昇格
    has_methods = any(c["section"] == "METHODS" for c in chunks)
    if not has_methods:
        for chunk in chunks:
            if chunk["section"] in FALLBACK_SECTIONS:
                chunk["is_primary"] = True

    return {
        "pxd": pxd_id,
        "file": txt_path,
        "chunks": chunks,
        "empty_sections": empty_sections,
        "has_methods": has_methods,
        "raw_files": raw_files,
    }


def get_llm_chunks(paper_result):
    """
    LLMに渡すべきチャンクだけをフィルタして返す。
    is_primary=True のチャンクのみ。
    """
    return [c for c in paper_result["chunks"] if c["is_primary"]]


def summarize(paper_result):
    """チャンク結果のサマリーを文字列で返す。"""
    lines = [f"PXD: {paper_result['pxd']}"]
    lines.append(f"Raw files: {len(paper_result['raw_files'])}")
    lines.append(f"Has METHODS: {paper_result['has_methods']}")
    lines.append(f"Empty sections: {', '.join(paper_result['empty_sections']) or 'none'}")
    lines.append("")
    lines.append(f"{'Section':<20} {'Primary':>8} {'Chars':>8}")
    lines.append("-" * 40)
    for c in paper_result["chunks"]:
        primary = "YES" if c["is_primary"] else "no"
        lines.append(f"{c['section']:<20} {primary:>8} {c['char_count']:>8}")
    return "\n".join(lines)


# CLI: ディレクトリを指定して全ファイルを処理
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk_pubtext.py <pubtext_dir>")
        sys.exit(1)

    pubtext_dir = sys.argv[1]
    txt_files = sorted(glob.glob(os.path.join(pubtext_dir, "*_PubText.txt")))

    if not txt_files:
        print(f"No *_PubText.txt files found in {pubtext_dir}")
        sys.exit(1)

    all_results = []
    for txt_file in txt_files:
        result = chunk_paper(txt_file)
        all_results.append(result)
        print(summarize(result))
        print()

    # 統計
    print("=" * 50)
    print(f"Total papers: {len(all_results)}")
    no_methods = [r for r in all_results if not r["has_methods"]]
    print(f"Papers without METHODS: {len(no_methods)}")
    if no_methods:
        for r in no_methods:
            print(f"  - {r['pxd']}")

    # LLMに渡すチャンク数の統計
    total_chunks = sum(len(get_llm_chunks(r)) for r in all_results)
    print(f"Total LLM chunks: {total_chunks}")
    print(f"Average chunks per paper: {total_chunks / len(all_results):.1f}")
