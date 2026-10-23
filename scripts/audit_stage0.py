"""V4 Stage 0 audit: re-check claimed statuses against the repository and write evidence.

  python scripts/audit_stage0.py      -> outputs/audit/stage0_audit.json (+ printed summary)

Checks models/locks, dataset counts/hashes/leakage, run storage, upload formats and hardware selection.
It does not train or run benchmarks. Dataset rebuild: scripts/build_datasets.py --out <tmp>.
Checkpoint re-evaluation: scripts/eval_checkpoint.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fairlens.config as cfgmod  # noqa: E402

OUT = ROOT / "outputs" / "audit"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def audit_models() -> dict:
    ver = json.loads((ROOT / "outputs/model_verification/verification.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "configs/model_lock.json").read_text(encoding="utf-8"))
    hub = Path(os.environ["HF_HOME"]) / "hub"
    rows = []
    for k, r in ver.items():
        mid = r.get("model_id")
        snap = hub / ("models--" + mid.replace("/", "--")) / "snapshots"
        have = sorted(p.name for p in snap.iterdir()) if snap.exists() else []
        rows.append({"key": k, "model_id": mid, "status": r.get("status"), "verified_at": r.get("verified_at"),
                     "sha": r.get("sha"), "lock_sha": lock.get(mid, {}).get("sha"),
                     "lock_matches": lock.get(mid, {}).get("sha") == r.get("sha"),
                     "pinned_snapshot_in_cache": r.get("sha") in have, "sample_output": r.get("sample_output")})
    cfg = cfgmod.load_config()
    used = {cfg["detectors"]["bart_mnli"]["model_id"], cfg["detectors"]["bart_mnli"]["lite_model_id"],
            cfg["detectors"]["hub_pretrained"]["model_id"], cfg["detectors"]["bert"]["base_model"],
            cfg["detectors"]["deberta"]["base_model"], cfg["preservation"]["embedding_model"], cfg["preservation"]["nli_model"],
            cfg["captioning"]["alignment_model"], cfg["captioning"]["captioners"]["blip"]["model_id"],
            cfg["captioning"]["captioners"]["florence"]["model_id"], cfg["captioning"]["captioners"]["florence"]["fallback_model_id"]}
    for r in cfg["mitigation"]["rewriters"].values():
        used |= {r.get(k) for k in ("model_id", "lite_model_id", "large_model_id") if r.get(k)}
    return {"n_records": len(rows), "n_verified": sum(r["status"] == "verified" for r in rows), "n_locked": len(lock),
            "all_lock_match": all(r["lock_matches"] for r in rows), "all_cached": all(r["pinned_snapshot_in_cache"] for r in rows),
            "config_models_without_lock": sorted(m for m in used if m not in lock), "models": rows}


def audit_datasets() -> dict:
    import pandas as pd

    man = json.loads((ROOT / "datasets/processed/dataset_manifest.json").read_text(encoding="utf-8"))
    files = {}
    for name, rec in man["files"].items():
        p = ROOT / "datasets/processed" / name
        files[name] = {"manifest_rows": rec["rows"], "actual_rows": len(pd.read_parquet(p)), "manifest_sha16": rec["sha256"],
                       "actual_sha16": sha16(p)}
        files[name]["ok"] = files[name]["manifest_rows"] == files[name]["actual_rows"] and rec["sha256"] == files[name]["actual_sha16"]
    det = pd.read_parquet(ROOT / "datasets/processed/detection.parquet")
    norm = det.assign(n=det.text.str.lower().str.replace(r"\W+", " ", regex=True).str.strip())
    other = {}
    for name in ["mitigation_eval.parquet", "gate_controlled.parquet"]:
        p = ROOT / "datasets/processed" / name
        other[name] = {"rows": len(pd.read_parquet(p)), "sha16": sha16(p), "in_manifest": name in man["files"]}
    mit = pd.read_parquet(ROOT / "datasets/processed/mitigation_eval.parquet")
    gate = pd.read_parquet(ROOT / "datasets/processed/gate_controlled.parquet")
    return {"files": files,
            "group_ids_spanning_splits": int((det.groupby("group_id")["split"].nunique() > 1).sum()),
            "exact_texts_spanning_splits": int((det.groupby("text")["split"].nunique() > 1).sum()),
            "normalised_texts_spanning_splits": int((norm.groupby("n")["split"].nunique() > 1).sum()),
            "split_label_counts": {f"{s}/{l}": int(n) for (s, l), n in det.groupby(["split", "label"]).size().items()},
            "other_files": other,
            "mitigation_benchmark_items_in_train_or_val": int(mit[mit.source != "authored"].text.isin(det[det.split != "test"].text).sum()),
            "gate_inputs_spanning_val_test": int((gate.groupby(["seed", "input_id"])["split"].nunique() > 1).sum()),
            "authored_cases_rows": sum(1 for _ in open(ROOT / "datasets/curated/authored_cases.jsonl", encoding="utf-8"))}


def audit_runs() -> dict:
    rows = []
    with sqlite3.connect(ROOT / "outputs/runs.sqlite") as con:
        for r in con.execute("SELECT run_id, kind, status, started_at, finished_at, manifest_path FROM runs ORDER BY started_at"):
            mp = Path(r[5])
            man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            rows.append({"run_id": r[0], "kind": r[1], "status_db": r[2], "status_manifest": man.get("status"),
                         "finished_at": r[4], "manifest_exists": mp.exists(), "manifest_path_stored": r[5],
                         "manifest_path_is_absolute": mp.is_absolute(), "config": man.get("config"),
                         "manifest_fields": sorted(man)})
    folders = sorted(p.name for p in (ROOT / "outputs/runs").iterdir() if p.is_dir())
    ckpts = {d.name: {"files": sorted(p.name for p in d.iterdir()),
                      "has_weights": any(p.suffix in {".safetensors", ".bin"} for p in d.iterdir())}
             for d in (ROOT / "checkpoints").iterdir() if d.is_dir()}
    return {"runs": rows, "run_folders": folders, "benchmark_runs": [r["run_id"] for r in rows if r["kind"].startswith("benchmark")],
            "checkpoints": ckpts}


def audit_uploads() -> dict:
    from docx import Document
    from PIL import Image
    from reportlab.pdfgen import canvas

    from fairlens.ingestion.loaders import IngestionError, extract_text, load_image

    text = "Looking for a young, aggressive salesman with 5 years of experience."
    b = {"txt": text.encode(), "csv": ("id,text\n1," + text + "\n").encode()}
    d = Document(); d.add_paragraph(text); buf = io.BytesIO(); d.save(buf); b["docx"] = buf.getvalue()
    buf = io.BytesIO(); c = canvas.Canvas(buf); c.drawString(72, 720, text); c.save(); b["pdf"] = buf.getvalue()
    res = {}
    for ext, data in b.items():
        try:
            doc = extract_text(f"sample.{ext}", data)
            res[ext] = {"ok": text in doc.text.replace("\n", " "), "chars": len(doc.text)}
        except IngestionError as e:
            res[ext] = {"ok": False, "error": str(e)}
    for ext, fmt in [("jpg", "JPEG"), ("png", "PNG"), ("gif", "GIF")]:
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), "red").save(buf, format=fmt)
        try:
            img = load_image(f"sample.{ext}", buf.getvalue())
            res[ext] = {"ok": img.size == (64, 48) and img.mode == "RGB"}
        except IngestionError as e:
            res[ext] = {"ok": False, "error": str(e)}
    # Windows-saved text and DOCX tables (V4 1.5 edge cases)
    edge = {}
    for name, data in [("utf16_crlf.txt", ("Line one.\r\n" + text + "\r\n").encode("utf-16")),
                       ("cp1252.txt", ("Café résumé: " + text).encode("cp1252"))]:
        try:
            edge[name] = {"ok": text in extract_text(name, data).text}
        except IngestionError as e:
            edge[name] = {"ok": False, "error": str(e)}
    d = Document(); t = d.add_table(rows=1, cols=2); t.cell(0, 0).text = "Requirement"; t.cell(0, 1).text = text
    buf = io.BytesIO(); d.save(buf)
    try:
        edge["docx_table"] = {"ok": text in extract_text("t.docx", buf.getvalue()).text}
    except IngestionError as e:
        edge["docx_table"] = {"ok": False, "error": str(e)}
    neg = {}
    for name, data in [("empty.txt", b""), ("bad.pdf", b"not a pdf"), ("x.exe", b"MZ"), ("bad.png", b"garbage")]:
        try:
            (load_image if name.endswith(".png") else extract_text)(name, data)
            neg[name] = "accepted (unexpected)"
        except IngestionError as e:
            neg[name] = f"rejected: {e}"
    return {"function_level": res, "edge_cases": edge, "negative_cases": neg,
            "note": "Function level only; the browser end-to-end check is V4 step 1.5."}


def audit_hardware() -> dict:
    from fairlens import hardware

    src = "".join((ROOT / "src/fairlens" / p).read_text(encoding="utf-8") for p in
                  ["hardware.py", "models.py", "detection/hf_classifier.py", "mitigation/llm_adapters.py", "multimodal/captioners.py",
                   "evaluation/preservation.py", "multimodal/caption_evaluation.py", "ui/state.py"])
    info = hardware.detect(check_internet=False)
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    return {"platform": platform.platform(), "detected": info.to_dict(), "profile": hardware.select_profile(info),
            "device()": hardware.device(), "checks_cuda": "cuda" in src, "checks_mps": "mps" in src,
            "FAIRLENS_DEVICE_override": "FAIRLENS_DEVICE" in src, "models_moved_to_device": ".to(" in src,
            "hardcoded_cpu_sentence_transformer": 'device="cpu"' in src, "topbar_hardcodes_CPU": "CPU · {prof}" in app}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
              "platform": platform.platform()}
    for name, fn in [("models", audit_models), ("datasets", audit_datasets), ("runs", audit_runs),
                     ("uploads", audit_uploads), ("hardware", audit_hardware)]:
        try:
            report[name] = fn()
        except Exception as exc:  # noqa: BLE001
            report[name] = {"audit_error": f"{type(exc).__name__}: {exc}"}
    (OUT / "stage0_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    m, d, r, u, h = (report[k] for k in ["models", "datasets", "runs", "uploads", "hardware"])
    print("models:", {k: m.get(k) for k in ["n_records", "n_verified", "n_locked", "all_lock_match", "all_cached", "config_models_without_lock"]})
    print("datasets:", {k: v for k, v in d.items() if k not in {"files", "split_label_counts", "other_files"}})
    print("  files ok:", {k: v["ok"] for k, v in d.get("files", {}).items()}, "| other:", d.get("other_files"))
    print("runs:", [(x["run_id"], x["status_db"], x["status_manifest"], x["manifest_path_is_absolute"]) for x in r.get("runs", [])])
    print("  benchmark runs:", r.get("benchmark_runs"), "| checkpoints:", {k: v["has_weights"] for k, v in r.get("checkpoints", {}).items()})
    print("uploads:", json.dumps(u))
    print("hardware:", {k: v for k, v in h.items() if k != "detected"})


if __name__ == "__main__":
    main()
