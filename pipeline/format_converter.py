"""
Step 5: LLM抽出結果をSDRF形式に変換する。
- 自然言語 → UNIMOD/PSI-MS形式
- 単位変換 (mmu → Da 等)

Usage:
    from format_converter import convert_extraction
    converted = convert_extraction(extraction_json)
"""
import re
import json
import os
import sys
import glob

# === CleavageAgent 変換テーブル ===
CLEAVAGE_AGENTS = {
    "trypsin": "AC=MS:1001251;NT=Trypsin",
    "trypsin/p": "AC=MS:1001313;NT=Trypsin/P",
    "lys-c": "AC=MS:1001309;NT=Lys-C",
    "lysc": "AC=MS:1001309;NT=Lys-C",
    "arg-c": "AC=MS:1001303;NT=Arg-C",
    "asp-n": "AC=MS:1001304;NT=Asp-N",
    "chymotrypsin": "AC=MS:1001306;NT=Chymotrypsin",
    "glu-c": "AC=MS:1001917;NT=Glu-C",
    "gluc": "AC=MS:1001917;NT=Glu-C",
    "pepsin": "AC=MS:1001311;NT=Pepsin",
    "proteinase k": "AC=MS:1001308;NT=Proteinase K",
    "cnbr": "AC=MS:1001307;NT=CNBr",
    "no cleavage": "AC=MS:1001955;NT=No cleavage",
    "no enzyme": "AC=MS:1001955;NT=No cleavage",
    "unspecific cleavage": "AC=MS:1001956;NT=unspecific cleavage",
}

# === Instrument 変換テーブル ===
INSTRUMENTS = {
    "q exactive": "AC=MS:1001911;NT=Q Exactive",
    "q exactive hf": "AC=MS:1002523;NT=Q Exactive HF",
    "q exactive hf-x": "AC=MS:1002877;NT=Q Exactive HF-X",
    "q exactive plus": "AC=MS:1002634;NT=Q Exactive Plus",
    "orbitrap fusion": "AC=MS:1002416;NT=Orbitrap Fusion",
    "orbitrap fusion lumos": "AC=MS:1002732;NT=Orbitrap Fusion Lumos",
    "orbitrap fusion etd": "AC=MS:1002417;NT=Orbitrap Fusion ETD",
    "orbitrap exploris 480": "AC=MS:1003028;NT=Orbitrap Exploris 480",
    "orbitrap exploris 120": "AC=MS:1003096;NT=Orbitrap Exploris 120",
    "orbitrap astral": "AC=MS:1003378;NT=Orbitrap Astral",
    "orbitrap elite": "AC=MS:1001910;NT=LTQ Orbitrap Elite",
    "ltq orbitrap elite": "AC=MS:1001910;NT=LTQ Orbitrap Elite",
    "ltq orbitrap velos": "AC=MS:1001742;NT=LTQ Orbitrap Velos",
    "ltq orbitrap": "AC=MS:1000449;NT=LTQ Orbitrap",
    "ltq orbitrap xl": "AC=MS:1000556;NT=LTQ Orbitrap XL",
    "tims tof pro": "AC=MS:1003005;NT=timsTOF Pro",
    "timstof pro": "AC=MS:1003005;NT=timsTOF Pro",
    "timstof pro 2": "AC=MS:1003230;NT=timsTOF Pro 2",
    "tims tof pro 2": "AC=MS:1003230;NT=timsTOF Pro 2",
    "tripletof 5600": "AC=MS:1000932;NT=TripleTOF 5600",
    "tripletof 6600": "AC=MS:1002533;NT=TripleTOF 6600",
    "zenotof 7600": "AC=MS:1003293;NT=ZenoTOF 7600",
    "synapt ms": "AC=MS:1001782;NT=Synapt MS",
    "exactive": "AC=MS:1000649;NT=Exactive",
    "velos pro": "AC=MS:1001742;NT=LTQ Orbitrap Velos",
}

# === FragmentationMethod 変換テーブル ===
FRAGMENTATION_METHODS = {
    "hcd": "AC=MS:1000422;NT=HCD",
    "cid": "AC=MS:1000133;NT=CID",
    "etd": "AC=MS:1000598;NT=ETD",
    "ethcd": "AC=MS:1002631;NT=EThcD",
    "etcid": "AC=MS:1003247;NT=ETciD",
    "uvpd": "AC=MS:1003246;NT=UVPD",
}

# === Label 変換テーブル ===
LABELS = {
    "label free": "AC=MS:1002038;NT=label free sample",
    "label free sample": "AC=MS:1002038;NT=label free sample",
    "label-free": "AC=MS:1002038;NT=label free sample",
    "tmt": "AC=MS:1002616;NT=TMT",
    "tmt6plex": "AC=MS:1002616;NT=TMT6plex",
    "tmt10plex": "AC=MS:1002617;NT=TMT10plex",
    "tmt11plex": "AC=MS:1002812;NT=TMT11plex",
    "tmt16plex": "AC=MS:1003230;NT=TMTpro 16plex",
    "tmtpro": "AC=MS:1003230;NT=TMTpro 16plex",
    "itraq": "AC=MS:1002010;NT=iTRAQ",
    "itraq4plex": "AC=MS:1002010;NT=iTRAQ4plex",
    "itraq8plex": "AC=MS:1002011;NT=iTRAQ8plex",
    "silac light": "AC=MS:1002038;NT=SILAC light",
    "silac heavy": "AC=MS:1002038;NT=SILAC heavy",
    "silac medium": "AC=MS:1002038;NT=SILAC medium",
}

# === Modification 変換テーブル ===
MODIFICATIONS = {
    # key: 自然言語パターン(lowercase), value: SDRF形式
    "carbamidomethyl": {"nt": "Carbamidomethyl", "ac": "UNIMOD:4", "ta": "C", "mt": "Fixed"},
    "carbamidomethylation": {"nt": "Carbamidomethyl", "ac": "UNIMOD:4", "ta": "C", "mt": "Fixed"},
    "iodoacetamide": {"nt": "Carbamidomethyl", "ac": "UNIMOD:4", "ta": "C", "mt": "Fixed"},
    "oxidation": {"nt": "Oxidation", "ac": "UNIMOD:35", "ta": "M", "mt": "Variable"},
    "oxidized met": {"nt": "Oxidation", "ac": "UNIMOD:35", "ta": "M", "mt": "Variable"},
    "acetyl": {"nt": "Acetyl", "ac": "UNIMOD:1", "ta": "", "mt": "Variable", "pp": "Protein N-term"},
    "acetylation": {"nt": "Acetyl", "ac": "UNIMOD:1", "ta": "", "mt": "Variable", "pp": "Protein N-term"},
    "phospho": {"nt": "Phospho", "ac": "UNIMOD:21", "ta": "S,T,Y", "mt": "Variable"},
    "phosphorylation": {"nt": "Phospho", "ac": "UNIMOD:21", "ta": "S,T,Y", "mt": "Variable"},
    "deamidated": {"nt": "Deamidated", "ac": "UNIMOD:7", "ta": "N,Q", "mt": "Variable"},
    "deamidation": {"nt": "Deamidated", "ac": "UNIMOD:7", "ta": "N,Q", "mt": "Variable"},
    "glu->pyro-glu": {"nt": "Glu->pyro-Glu", "ac": "UNIMOD:27", "ta": "E", "mt": "Variable", "pp": "Any N-term"},
    "gln->pyro-glu": {"nt": "Gln->pyro-Glu", "ac": "UNIMOD:28", "ta": "Q", "mt": "Variable", "pp": "Any N-term"},
    "pyro-glu": {"nt": "Gln->pyro-Glu", "ac": "UNIMOD:28", "ta": "Q", "mt": "Variable", "pp": "Any N-term"},
    "methyl": {"nt": "Methyl", "ac": "UNIMOD:34", "ta": "K,R", "mt": "Variable"},
    "methylation": {"nt": "Methyl", "ac": "UNIMOD:34", "ta": "K,R", "mt": "Variable"},
    "dimethyl": {"nt": "Dimethyl", "ac": "UNIMOD:36", "ta": "K,R", "mt": "Variable"},
    "trimethyl": {"nt": "Trimethyl", "ac": "UNIMOD:37", "ta": "K", "mt": "Variable"},
    "ubiquitin": {"nt": "GlyGly", "ac": "UNIMOD:121", "ta": "K", "mt": "Variable"},
    "glygly": {"nt": "GlyGly", "ac": "UNIMOD:121", "ta": "K", "mt": "Variable"},
    "tmt6plex": {"nt": "TMT6plex", "ac": "UNIMOD:737", "ta": "K", "mt": "Fixed"},
    "tmt10plex": {"nt": "TMT6plex", "ac": "UNIMOD:737", "ta": "K", "mt": "Fixed"},
    "tmtpro": {"nt": "TMTpro", "ac": "UNIMOD:2016", "ta": "K", "mt": "Fixed"},
    "propionamide": {"nt": "Propionamide", "ac": "UNIMOD:24", "ta": "C", "mt": "Variable"},
    "carbamyl": {"nt": "Carbamyl", "ac": "UNIMOD:5", "ta": "", "mt": "Variable", "pp": "Peptide N-term"},
    "ammonia-loss": {"nt": "Ammonia-loss", "ac": "UNIMOD:385", "ta": "N", "mt": "Variable"},
    "ammonium": {"nt": "Ammonium", "ac": "UNIMOD:989", "ta": "D,E", "mt": "Variable"},
}


def convert_modification(raw_value):
    """自然言語のModification記述をSDRF形式に変換"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"

    # 既にUNIMOD形式なら返す
    if "UNIMOD:" in raw_value or "AC=" in raw_value:
        return raw_value

    raw_lower = raw_value.lower()

    # ターゲットアミノ酸の抽出 (括弧内)
    ta_match = re.search(r'\(([A-Z,\s]+)\)', raw_value)
    custom_ta = ta_match.group(1).replace(" ", "") if ta_match else None

    # Fixed/Variable 判定
    is_fixed = "fixed" in raw_lower or "static" in raw_lower

    # テーブルからマッチ
    for key, mod_info in MODIFICATIONS.items():
        if key in raw_lower:
            parts = [f"NT={mod_info['nt']}"]

            # ターゲットアミノ酸
            ta = custom_ta if custom_ta else mod_info.get("ta", "")
            if ta:
                parts.append(f"TA={ta}")

            parts.append(f"AC={mod_info['ac']}")

            # Position
            if mod_info.get("pp"):
                parts.append(f"PP={mod_info['pp']}")
            elif "n-term" in raw_lower or "n term" in raw_lower:
                parts.append("PP=Protein N-term")

            # Fixed/Variable
            mt = "Fixed" if is_fixed else mod_info.get("mt", "Variable")
            parts.append(f"MT={mt}")

            return ";".join(parts)

    # マッチしなかった場合はそのまま返す
    return raw_value


def convert_cleavage_agent(raw_value):
    """CleavageAgent を PSI-MS形式に変換"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"
    if "AC=" in raw_value:
        return raw_value

    key = raw_value.lower().strip()
    # 複数酵素の場合は最初のものだけ（カンマやスラッシュ区切り）
    for sep in [",", "/", ";", " and "]:
        if sep in key:
            key = key.split(sep)[0].strip()
            break

    return CLEAVAGE_AGENTS.get(key, raw_value)


def convert_instrument(raw_value):
    """Instrument を PSI-MS形式に変換"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"
    if "AC=" in raw_value:
        return raw_value

    key = raw_value.lower().strip()
    return INSTRUMENTS.get(key, raw_value)


def convert_fragmentation(raw_value):
    """FragmentationMethod を PSI-MS形式に変換"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"
    if "AC=" in raw_value:
        return raw_value

    # 複数メソッドの場合（CID;ETD 等）は最初のもの
    key = raw_value.lower().strip()
    for sep in [";", ",", "/", " and "]:
        if sep in key:
            key = key.split(sep)[0].strip()
            break

    return FRAGMENTATION_METHODS.get(key, raw_value)


def convert_label(raw_value):
    """Label を PSI-MS形式に変換"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"
    if "AC=" in raw_value:
        return raw_value

    key = raw_value.lower().strip()
    return LABELS.get(key, raw_value)


def convert_mass_tolerance(raw_value):
    """質量許容誤差の単位正規化 (mmu → Da, etc.)"""
    if not raw_value or raw_value.lower() in ("not applicable", ""):
        return "Not Applicable"

    raw_lower = raw_value.lower().strip()

    # mmu → Da 変換
    mmu_match = re.match(r'[±]?\s*([\d.]+)\s*mmu', raw_lower)
    if mmu_match:
        val = float(mmu_match.group(1))
        da_val = val / 1000
        # 末尾の不要な0を除去
        da_str = f"{da_val:.4f}".rstrip("0").rstrip(".")
        return f"{da_str} Da"

    # ± 記号の正規化
    raw_lower = raw_lower.replace("±", "").strip()

    # ppm はそのまま
    ppm_match = re.match(r'([\d.]+)\s*ppm', raw_lower)
    if ppm_match:
        return f"{ppm_match.group(1)} ppm"

    # Da はそのまま
    da_match = re.match(r'([\d.]+)\s*da', raw_lower)
    if da_match:
        return f"{da_match.group(1)} Da"

    return raw_value


def convert_extraction(extraction_data):
    """
    LLM抽出結果全体をSDRF形式に変換する。
    元のextraction JSONを受け取り、変換済みJSONを返す。
    """
    converted = json.loads(json.dumps(extraction_data))  # deep copy

    for group_name, group_data in converted.get("extractions", {}).items():
        for category, entry in group_data.items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue

            raw_value = entry["value"]
            original = raw_value

            # カテゴリ別の変換
            if "CleavageAgent" in category:
                entry["value"] = convert_cleavage_agent(raw_value)
            elif "Instrument" in category:
                entry["value"] = convert_instrument(raw_value)
            elif "FragmentationMethod" in category:
                entry["value"] = convert_fragmentation(raw_value)
            elif "Label" in category:
                entry["value"] = convert_label(raw_value)
            elif "Modification" in category and "Number" not in category:
                entry["value"] = convert_modification(raw_value)
            elif "MassTolerance" in category:
                entry["value"] = convert_mass_tolerance(raw_value)

            # 変換が行われた場合は記録
            if entry["value"] != original:
                entry["original_value"] = original
                entry["converted"] = True

    return converted


def convert_extraction_v2(extraction_v2_data):
    """
    v2形式（steps構造）のLLM抽出結果をSDRF形式に変換する。
    """
    converted = json.loads(json.dumps(extraction_v2_data))  # deep copy

    for step_name, step_data in converted.get("steps", {}).items():
        if not isinstance(step_data, dict):
            continue
        for section in ["confirmed", "hypotheses"]:
            for category, entry in step_data.get(section, {}).items():
                if not isinstance(entry, dict):
                    continue

                # hypothesesの場合、最初の候補をvalueにする
                raw_value = entry.get("value", "")
                if not raw_value and "candidates" in entry:
                    candidates = entry["candidates"]
                    raw_value = candidates[0] if candidates else ""
                    entry["value"] = raw_value

                if not raw_value:
                    continue

                original = raw_value
                if "CleavageAgent" in category:
                    entry["value"] = convert_cleavage_agent(raw_value)
                elif "Instrument" in category:
                    entry["value"] = convert_instrument(raw_value)
                elif "FragmentationMethod" in category:
                    entry["value"] = convert_fragmentation(raw_value)
                elif "Label" in category:
                    entry["value"] = convert_label(raw_value)
                elif "Modification" in category and "Number" not in category:
                    entry["value"] = convert_modification(raw_value)
                elif "MassTolerance" in category:
                    entry["value"] = convert_mass_tolerance(raw_value)

                if entry["value"] != original:
                    entry["original_value"] = original
                    entry["converted"] = True

    return converted


def sort_modifications(mod_values):
    """
    Modification値をFixed→Variable順にソートする。
    入力: ["NT=Oxidation;...;MT=Variable", "NT=Carbamidomethyl;...;MT=Fixed", ...]
    出力: Fixed が先、Variable が後
    """
    fixed = []
    variable = []
    other = []
    for val in mod_values:
        if not val or val == "Not Applicable":
            continue
        val_lower = val.lower()
        if "mt=fixed" in val_lower:
            fixed.append(val)
        elif "mt=variable" in val_lower:
            variable.append(val)
        else:
            other.append(val)
    return fixed + variable + other


def drop_low_confidence_hypotheses(extraction_v2_data):
    """
    v2形式のhypothesisでconfidence=lowのものをNot Applicableに落とす。
    確定できない推論は出さない方が安全。
    """
    result = json.loads(json.dumps(extraction_v2_data))

    for step_name, step_data in result.get("steps", {}).items():
        if not isinstance(step_data, dict):
            continue
        hypotheses = step_data.get("hypotheses", {})
        to_remove = []
        for category, entry in hypotheses.items():
            if isinstance(entry, dict):
                conf = entry.get("confidence", "low")
                if conf == "low":
                    to_remove.append(category)

        for cat in to_remove:
            del hypotheses[cat]
            na_list = step_data.get("not_applicable", [])
            if cat not in na_list:
                na_list.append(cat)
                step_data["not_applicable"] = na_list

    return result


# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_converter.py <extraction_json> or <extraction_dir>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*_extraction.json")))
    else:
        files = [target]

    output_dir = os.path.join(os.path.dirname(files[0]), "converted")
    os.makedirs(output_dir, exist_ok=True)

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        converted = convert_extraction(data)
        pxd = converted.get("pxd", "unknown")

        out_file = os.path.join(output_dir, f"{pxd}_converted.json")
        with open(out_file, "w") as fh:
            json.dump(converted, fh, indent=2, ensure_ascii=False)

        # 変換サマリー
        conv_count = 0
        for group_data in converted.get("extractions", {}).values():
            for entry in group_data.values():
                if isinstance(entry, dict) and entry.get("converted"):
                    conv_count += 1
        print(f"{pxd}: {conv_count} fields converted → {out_file}")
