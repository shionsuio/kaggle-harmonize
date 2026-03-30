"""
寝てる間に訓練データ50件をLLM抽出する。
レートリミット対応リトライ + PRIDE anchor自動取得付き。
"""
import os, sys, glob, time, json

sys.path.insert(0, os.path.dirname(__file__))
from extract_with_llm_v2 import extract_paper_v2
from fetch_pride_anchors import extract_anchors

import anthropic

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    output_dir = os.path.join(BASE_DIR, "pipeline", "llm_extractions")
    anchor_dir = os.path.join(BASE_DIR, "pipeline", "pride_anchors")

    # 既にv2抽出済みのPXD
    done = set()
    for f in glob.glob(os.path.join(output_dir, "*_extraction_v2.json")):
        pxd = os.path.basename(f).replace("_extraction_v2.json", "")
        # エラー抽出を除外
        with open(f) as fh:
            ext = json.load(fh)
        has_data = any(
            isinstance(ext.get("steps", {}).get(s, {}), dict) and
            ext["steps"][s].get("confirmed")
            for s in ["STEP1_OVERVIEW", "STEP2_SAMPLE_PREP", "STEP3_MS_SETTINGS"]
        )
        if has_data:
            done.add(pxd)

    # 訓練PXDで未抽出のもの（最大50件）
    todo = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "Training_SDRFs", "HarmonizedFiles", "Harmonized_PXD*.csv"))):
        pxd = os.path.basename(f).replace("Harmonized_", "").replace(".csv", "")
        if pxd not in done:
            todo.append(pxd)
    todo = todo[:50]

    print(f"Already done: {len(done)}")
    print(f"To process: {len(todo)}")
    print(f"Papers: {todo}")
    print()

    success = 0
    fail = 0

    for i, pxd in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] {pxd}")

        # PubText確認
        pubtext_path = os.path.join(BASE_DIR, "Training_PubText", "PubText", f"{pxd}_PubText.txt")
        if not os.path.exists(pubtext_path):
            print(f"  SKIP: no PubText file")
            fail += 1
            continue

        # PRIDE anchor取得（なければ自動取得）
        anchor_path = os.path.join(anchor_dir, f"{pxd}_anchor.json")
        if not os.path.exists(anchor_path):
            print(f"  Fetching PRIDE anchor...", end=" ")
            try:
                anchor = extract_anchors(pxd)
                if anchor:
                    with open(anchor_path, "w") as f:
                        json.dump(anchor, f, indent=2, ensure_ascii=False)
                    print("OK")
                else:
                    print("FAILED")
                    fail += 1
                    continue
            except Exception as e:
                print(f"ERROR: {e}")
                fail += 1
                continue
            time.sleep(0.5)

        # LLM抽出
        try:
            extract_paper_v2(client, pxd, pubtext_path, anchor_path, output_dir)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1

        # レートリミット対策：各論文間で少し待つ
        time.sleep(5)

    print(f"\n{'='*60}")
    print(f"Done. Success: {success}, Failed: {fail}")

if __name__ == "__main__":
    main()
