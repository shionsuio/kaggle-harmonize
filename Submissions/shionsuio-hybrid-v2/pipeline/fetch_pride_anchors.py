"""
Step 1: PRIDE APIからアンカー情報を取得し、パイプライン用JSONに整形保存する。
Usage: python fetch_pride_anchors.py <PXD_ID or directory_of_pubtext>
"""
import json, os, sys, re, time
import urllib.request

PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{}"
PRIDE_FILES_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{}/files"

# Instrument → 連鎖推論テーブル
INSTRUMENT_LOOKUP = {
    "Q Exactive": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Q Exactive HF": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Q Exactive HF-X": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Q Exactive Plus": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Orbitrap Exploris 480": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Orbitrap Exploris 120": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Orbitrap Astral": {"fragmentation": "HCD", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Orbitrap Fusion": {"fragmentation": None, "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},  # HCD/CID/ETD possible
    "Orbitrap Fusion Lumos": {"fragmentation": None, "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "Orbitrap Fusion ETD": {"fragmentation": None, "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "LTQ Orbitrap": {"fragmentation": "CID", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "LTQ Orbitrap Velos": {"fragmentation": None, "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "LTQ Orbitrap Elite": {"fragmentation": None, "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "LTQ Orbitrap XL": {"fragmentation": "CID", "ms2_analyzer": "orbitrap", "ionization": "nanoESI"},
    "TripleTOF 5600": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "nanoESI"},
    "TripleTOF 6600": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "nanoESI"},
    "ZenoTOF 7600": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "nanoESI"},
    "Synapt MS": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "ESI"},
    "timsTOF Pro": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "nanoESI"},
    "timsTOF Pro 2": {"fragmentation": "CID", "ms2_analyzer": "TOF", "ionization": "nanoESI"},
}

def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  WARN: Failed to fetch {url}: {e}")
        return None

def extract_anchors(pxd_id):
    """PRIDE APIからアンカー情報を抽出"""
    anchor = {
        "pxd": pxd_id,
        "source": "PRIDE_API",
        "organism": [],
        "instruments": [],
        "modifications": [],
        "organism_parts": [],
        "diseases": [],
        "experiment_types": [],
        "quantification_methods": [],
        "keywords": [],
        "publications": [],
        # 連鎖推論
        "inferred_fragmentation": None,
        "inferred_ms2_analyzer": None,
        "inferred_ionization": None,
        # rawファイル一覧
        "raw_files": [],
    }

    # Project metadata
    data = fetch_json(PRIDE_API.format(pxd_id))
    if not data:
        return None

    # Organisms
    for o in data.get("organisms", []):
        anchor["organism"].append({
            "name": o.get("name", ""),
            "accession": o.get("accession", ""),
        })

    # Instruments
    for i in data.get("instruments", []):
        name = i.get("name", "")
        acc = i.get("accession", "")
        anchor["instruments"].append({
            "name": name,
            "accession": acc,
            "sdrf_format": f"AC={acc};NT={name}" if acc and name else "",
        })
        # 連鎖推論
        if name in INSTRUMENT_LOOKUP:
            lookup = INSTRUMENT_LOOKUP[name]
            if lookup["fragmentation"] and not anchor["inferred_fragmentation"]:
                anchor["inferred_fragmentation"] = lookup["fragmentation"]
            if lookup["ms2_analyzer"] and not anchor["inferred_ms2_analyzer"]:
                anchor["inferred_ms2_analyzer"] = lookup["ms2_analyzer"]
            if lookup["ionization"] and not anchor["inferred_ionization"]:
                anchor["inferred_ionization"] = lookup["ionization"]

    # Modifications
    for m in data.get("ptmList", []):
        anchor["modifications"].append({
            "name": m.get("name", ""),
            "accession": m.get("accession", ""),
        })

    # Organism parts
    for op in data.get("organismParts", []):
        anchor["organism_parts"].append({
            "name": op.get("name", ""),
            "accession": op.get("accession", ""),
        })

    # Diseases
    for d in data.get("diseases", []):
        anchor["diseases"].append({
            "name": d.get("name", ""),
            "accession": d.get("accession", ""),
        })

    # Experiment types
    for et in data.get("experimentTypes", []):
        anchor["experiment_types"].append(et.get("name", ""))

    # Quantification methods
    for qm in data.get("quantificationMethods", []):
        anchor["quantification_methods"].append(qm.get("name", ""))

    # Keywords
    anchor["keywords"] = data.get("keywords", [])

    # Publications
    for pub in data.get("references", []):
        anchor["publications"].append({
            "pubmed_id": pub.get("pubmedId", ""),
            "doi": pub.get("doi", ""),
            "ref": pub.get("referenceLine", ""),
        })

    # Files
    files_data = fetch_json(PRIDE_FILES_API.format(pxd_id))
    if files_data and isinstance(files_data, list):
        for f in files_data:
            cat = f.get("fileCategory", {}).get("value", "")
            name = f.get("fileName", "")
            if cat == "RAW":
                anchor["raw_files"].append(name)

    return anchor


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_pride_anchors.py <PXD_ID> [PXD_ID2 ...] or <pubtext_dir>")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), "pride_anchors")
    os.makedirs(out_dir, exist_ok=True)

    pxd_ids = []
    if os.path.isdir(sys.argv[1]):
        # ディレクトリからPXD IDを抽出
        for fname in os.listdir(sys.argv[1]):
            m = re.match(r"(PXD\d+)_PubText", fname)
            if m:
                pxd_ids.append(m.group(1))
    else:
        pxd_ids = [a for a in sys.argv[1:] if a.startswith("PXD")]

    pxd_ids = sorted(set(pxd_ids))
    print(f"Fetching {len(pxd_ids)} projects from PRIDE API...")

    for i, pxd in enumerate(pxd_ids):
        print(f"[{i+1}/{len(pxd_ids)}] {pxd}...", end=" ")
        out_file = os.path.join(out_dir, f"{pxd}_anchor.json")

        if os.path.exists(out_file):
            print("cached")
            continue

        anchor = extract_anchors(pxd)
        if anchor:
            with open(out_file, "w") as f:
                json.dump(anchor, f, indent=2, ensure_ascii=False)
            print(f"OK ({len(anchor['raw_files'])} raw files)")
        else:
            print("FAILED")

        time.sleep(0.5)

    print(f"\nDone. Anchors saved to {out_dir}/")


if __name__ == "__main__":
    main()
