from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import re
from xml.dom import minidom
import xml.etree.ElementTree as ET

import pandas as pd

# Ensure repo root is on sys.path so `import src...` works when running scripts directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402

HREF_ID_RE = re.compile(r"id\(([^)]+)\)")


def _first_dir_with_xml(root: Path, target_dir_name: str) -> Path:
    """
    Find the first directory (case-insensitive name match) that contains at least one .xml file.
    Searches recursively under `root`.
    """
    name_lower = target_dir_name.lower()
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == name_lower:
            xmls = list(p.glob("*.xml"))
            if xmls:
                candidates.append((len(xmls), p))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a '{target_dir_name}' directory with XML files under: {root}\n"
            f"Your extracted AMI annotations are likely in a different layout."
        )
    # prefer the directory with more xml files
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_file(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find '{filename}' under: {root}")
    # prefer the shortest path (usually the canonical one)
    hits.sort(key=lambda p: len(p.parts))
    return hits[0]


def parse_href(href: str) -> tuple[str, str | None, str | None]:
    # Example:
    # ES2002a.B.words.xml#id(ES2002a.B.words4)..id(ES2002a.B.words16)
    if "#" not in href:
        return href, None, None
    filename, frag = href.split("#", 1)
    ids = HREF_ID_RE.findall(frag)
    if not ids:
        return filename, None, None
    if len(ids) == 1:
        return filename, ids[0], None
    return filename, ids[0], ids[1]


def _extract_pointer_id(href: str) -> str | None:
    # da-types.xml#id(ami_da_14)
    m = HREF_ID_RE.search(href)
    return m.group(1) if m else None


def load_da_types_map(da_types_xml: Path) -> dict[str, str]:
    """
    Build a mapping from ami_da_* ids to human-readable label strings.
    We do not assume a single schema; we try common attributes and fallback to id.
    """
    tree = ET.parse(str(da_types_xml))
    root = tree.getroot()

    id_to_label: dict[str, str] = {}
    for el in root.iter():
        # NXT typically uses nite:id
        nite_id = el.attrib.get("nite:id") or el.attrib.get("{http://nite.sourceforge.net/}id")
        if not nite_id:
            continue

        # try common attribute names
        label = (
            el.attrib.get("name")
            or el.attrib.get("gloss")
            or el.attrib.get("label")
            or el.attrib.get("abbr")
            or el.attrib.get("short")
        )

        # try text content if attributes absent
        if not label:
            txt = (el.text or "").strip()
            if txt:
                label = txt

        # final fallback
        if not label:
            label = nite_id

        id_to_label[nite_id] = label

    return id_to_label


def load_words(words_path: Path) -> tuple[list[str], list[bool], dict[str, int], list[float | None], list[float | None]]:
    doc = minidom.parse(str(words_path))
    nodes = doc.getElementsByTagName("w")

    tokens: list[str] = []
    is_punc: list[bool] = []
    start_times: list[float | None] = []
    end_times: list[float | None] = []
    id_to_pos: dict[str, int] = {}

    for n in nodes:
        nite_id = n.getAttribute("nite:id")
        txt = ""
        if n.firstChild is not None:
            txt = n.firstChild.data

        punc_flag = n.hasAttribute("punc")

        st = n.getAttribute("starttime") or ""
        et = n.getAttribute("endtime") or ""
        st_f = float(st) if st.strip() else None
        et_f = float(et) if et.strip() else None

        pos = len(tokens)
        tokens.append(txt)
        is_punc.append(punc_flag)
        start_times.append(st_f)
        end_times.append(et_f)

        if nite_id:
            id_to_pos[nite_id] = pos

    return tokens, is_punc, id_to_pos, start_times, end_times


def join_tokens(tokens: list[str], is_punc: list[bool]) -> str:
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out += t
            continue
        if is_punc[i]:
            out += t
        else:
            out += " " + t
    return out.strip()


def span_text(
    tokens: list[str],
    is_punc: list[bool],
    id_to_pos: dict[str, int],
    w_st: list[float | None],
    w_et: list[float | None],
    start_id: str | None,
    end_id: str | None,
    act_start: float | None,
    act_end: float | None,
) -> str:
    # Preferred: span by word ids
    if start_id and start_id in id_to_pos:
        a = id_to_pos[start_id]
        b = a
        if end_id and end_id in id_to_pos:
            b = id_to_pos[end_id]
        if b < a:
            a, b = b, a
        return join_tokens(tokens[a : b + 1], is_punc[a : b + 1])

    # Fallback: span by time window if ids missing
    if act_start is None or act_end is None:
        return ""

    idx = []
    for i, (st, et) in enumerate(zip(w_st, w_et)):
        if st is None and et is None:
            continue
        stv = st if st is not None else act_start
        etv = et if et is not None else act_end
        if stv >= act_start and etv <= act_end:
            idx.append(i)

    if not idx:
        return ""

    a, b = idx[0], idx[-1]
    return join_tokens(tokens[a : b + 1], is_punc[a : b + 1])


def extract_dialogue_acts(
    annotations_root: Path,
    dialogue_acts_dir: Path,
    words_dir: Path,
    da_types_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    words_cache: dict[str, tuple] = {}

    da_files = sorted(dialogue_acts_dir.glob("*.xml"))
    if not da_files:
        raise FileNotFoundError(f"No dialogue act xml files found in: {dialogue_acts_dir}")

    for da_path in da_files:
        # Often: ES2002a.B.dialog-act.xml or ES2002a.B.dialogueActs.xml, etc.
        parts = da_path.name.split(".")
        meeting = parts[0] if len(parts) > 0 else da_path.stem
        speaker = parts[1] if len(parts) > 1 else None

        doc = minidom.parse(str(da_path))
        dacts = doc.getElementsByTagName("dact")

        for d in dacts:
            da_id = d.getAttribute("nite:id") or ""

            # Label is usually referenced via pointer to da-types.xml, not as an attribute. :contentReference[oaicite:1]{index=1}
            label = ""
            pointer_id = None
            for c in d.childNodes:
                if getattr(c, "nodeType", None) == c.ELEMENT_NODE and c.nodeName.endswith("pointer"):
                    if c.hasAttribute("href"):
                        pointer_id = _extract_pointer_id(c.getAttribute("href"))
                        if pointer_id:
                            label = da_types_map.get(pointer_id, pointer_id)
                            break

            st = d.getAttribute("starttime") or ""
            et = d.getAttribute("endtime") or ""
            act_start = float(st) if st.strip() else None
            act_end = float(et) if et.strip() else None

            href = None
            for c in d.childNodes:
                if getattr(c, "nodeType", None) == c.ELEMENT_NODE and c.nodeName.endswith("child"):
                    if c.hasAttribute("href"):
                        href = c.getAttribute("href")
                        break

            words_file, start_id, end_id = ("", None, None)
            text = ""

            if href:
                words_file, start_id, end_id = parse_href(href)
                if words_file:
                    if words_file not in words_cache:
                        # words_file is usually a filename like ES2002a.B.words.xml
                        wp = words_dir / words_file
                        if not wp.exists():
                            # sometimes paths in href omit folders; try searching within words_dir
                            hits = list(words_dir.rglob(words_file))
                            if hits:
                                wp = hits[0]
                        if not wp.exists():
                            raise FileNotFoundError(
                                f"Missing words file referenced by DA:\n"
                                f"DA file: {da_path.name}\n"
                                f"Referenced: {words_file}\n"
                                f"Searched in: {words_dir}"
                            )
                        words_cache[words_file] = load_words(wp)

                    tokens, is_punc, id_to_pos, w_st, w_et = words_cache[words_file]
                    text = span_text(tokens, is_punc, id_to_pos, w_st, w_et, start_id, end_id, act_start, act_end)

            rows.append(
                {
                    "meeting": meeting,
                    "speaker": speaker,
                    "dact_id": da_id,
                    "start": act_start,
                    "end": act_end,
                    "label": label,
                    "label_id": pointer_id,
                    "text": text,
                    "dact_file": da_path.name,
                    "words_file": words_file,
                    "start_word_id": start_id,
                    "end_word_id": end_id,
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["label"].astype(str).str.len() > 0].copy()
    df = df[df["text"].astype(str).str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def make_splits(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float, test_frac: float) -> dict:
    meetings = sorted(df["meeting"].unique().tolist())

    import random
    rng = random.Random(seed)
    rng.shuffle(meetings)

    n = len(meetings)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    _ = test_frac  # remainder

    train = meetings[:n_train]
    val = meetings[n_train : n_train + n_val]
    test = meetings[n_train + n_val :]

    return {
        "train": train,
        "val": val,
        "test": test,
        "counts": {"meetings": n, "train": len(train), "val": len(val), "test": len(test)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    annotations_root = Path(cfg["paths"]["ami"]["annotations_dir"])

    # Auto-detect the real locations (works across AMI layouts)
    dialogue_acts_dir = _first_dir_with_xml(annotations_root, "dialogueActs")
    words_dir = _first_dir_with_xml(annotations_root, "words")
    da_types_xml = _find_file(annotations_root, "da-types.xml")

    print("Using AMI annotation paths:")
    print(f"  annotations_root : {annotations_root}")
    print(f"  dialogueActs     : {dialogue_acts_dir}")
    print(f"  words            : {words_dir}")
    print(f"  da-types.xml     : {da_types_xml}")

    da_types_map = load_da_types_map(da_types_xml)

    out_parquet = Path(cfg["outputs"]["utterances_parquet"])
    out_csv = Path(cfg["outputs"]["utterances_csv"])
    out_splits = Path(cfg["outputs"]["splits_json"])

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = extract_dialogue_acts(
        annotations_root=annotations_root,
        dialogue_acts_dir=dialogue_acts_dir,
        words_dir=words_dir,
        da_types_map=da_types_map,
    )

    splits_cfg = cfg["splits"]
    splits = make_splits(
        df,
        seed=seed,
        train_frac=float(splits_cfg["train_frac"]),
        val_frac=float(splits_cfg["val_frac"]),
        test_frac=float(splits_cfg["test_frac"]),
    )

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    out_splits.parent.mkdir(parents=True, exist_ok=True)
    out_splits.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"\nSaved utterances: {len(df):,}")
    print(f"Unique labels: {df['label'].nunique():,}")
    print("\nTop labels:")
    print(df["label"].value_counts().head(12))
    print(f"\nWrote: {out_parquet}")
    print(f"Wrote: {out_splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())