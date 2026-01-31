import argparse
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_limit(rows: list[dict[str, Any]], max_samples: Optional[int], seed: int) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_samples]


def _max_rows_hint(max_samples: Optional[int]) -> Optional[int]:
    if max_samples is None or max_samples <= 0:
        return None
    return min(max_samples * 50, 5000)


def _should_use_parquet_fallback(e: Exception) -> bool:
    msg = str(e)
    return ("Dataset scripts are no longer supported" in msg) or ("trust_remote_code" in msg)


def _letter_from_index(idx: int) -> str:
    return "ABCD"[idx]


def _letter_from_index_any(idx: int) -> str:
    return chr(ord("A") + idx)


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for key in ["text", "content", "value", "label", "name"]:
            v = x.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(x).strip()


def _load_preferred_split(
    dataset_name: str,
    dataset_config: Optional[str],
    preferred_splits: list[str],
    cache_dir: Optional[str],
    max_rows: Optional[int] = None,
):
    if load_dataset is None:
        return _load_preferred_split_via_datasets_server(dataset_name, dataset_config, preferred_splits, max_rows)
    last_error: Optional[Exception] = None
    for split in preferred_splits:
        try:
            return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir), split
        except ValueError as e:
            last_error = e
            if 'Unknown split "' in str(e):
                continue
            raise
        except RuntimeError as e:
            last_error = e
            if _should_use_parquet_fallback(e):
                break
            raise

    try:
        ds = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    except RuntimeError as e:
        last_error = e
        if _should_use_parquet_fallback(e):
            ds, split = _load_via_datasets_server_parquet(dataset_name, dataset_config, preferred_splits, cache_dir)
            if max_rows is not None:
                try:
                    ds = ds.select(range(min(len(ds), max_rows)))
                except Exception:
                    pass
            return ds, split
        raise
    splits = list(getattr(ds, "keys", lambda: [])())
    if splits:
        return ds[splits[0]], splits[0]
    raise RuntimeError(f"Failed to load dataset splits for {dataset_name}/{dataset_config}: {last_error}")


def _http_get_json(url: str, timeout_s: float = 60.0) -> dict[str, Any]:
    req = urllib.request.Request(url=url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _datasets_server_base() -> str:
    return os.getenv("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co").rstrip("/")


def _datasets_server_splits(dataset_name: str) -> list[dict[str, Any]]:
    base = _datasets_server_base()
    ds_param = urllib.parse.quote(dataset_name, safe="")
    url = f"{base}/splits?dataset={ds_param}"
    data = _http_get_json(url)
    splits = data.get("splits")
    if not isinstance(splits, list):
        return []
    return [x for x in splits if isinstance(x, dict)]


def _datasets_server_rows(
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_rows: Optional[int],
) -> list[dict[str, Any]]:
    base = _datasets_server_base()
    ds_param = urllib.parse.quote(dataset_name, safe="")
    cfg_param = urllib.parse.quote(dataset_config, safe="")
    split_param = urllib.parse.quote(split, safe="")

    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        remaining = None if max_rows is None else max_rows - len(out)
        if remaining is not None and remaining <= 0:
            break
        length = 100
        if remaining is not None:
            length = min(length, remaining)
        url = (
            f"{base}/rows?dataset={ds_param}&config={cfg_param}"
            f"&split={split_param}&offset={offset}&length={length}"
        )
        data = _http_get_json(url)
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        got = 0
        for item in rows:
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            if isinstance(row, dict):
                out.append(row)
                got += 1
        if got < length:
            break
        offset += length
    return out


def _load_preferred_split_via_datasets_server(
    dataset_name: str,
    dataset_config: Optional[str],
    preferred_splits: list[str],
    max_rows: Optional[int],
):
    splits = _datasets_server_splits(dataset_name)
    if not splits:
        raise RuntimeError(f"No splits found from datasets-server for {dataset_name}")

    configs: list[str] = []
    for s in splits:
        cfg = s.get("config")
        if isinstance(cfg, str) and cfg:
            configs.append(cfg)
    configs = sorted(set(configs))

    chosen_config: Optional[str] = None
    if dataset_config:
        if dataset_config in configs:
            chosen_config = dataset_config
        else:
            chosen_config = dataset_config
    else:
        chosen_config = configs[0] if configs else None
    if not chosen_config:
        raise RuntimeError(f"No dataset config found for {dataset_name}")

    available_splits = sorted(
        {
            str(s.get("split"))
            for s in splits
            if s.get("config") == chosen_config and isinstance(s.get("split"), str)
        }
    )
    if not available_splits:
        raise RuntimeError(f"No splits found for {dataset_name}/{chosen_config}")

    chosen_split: Optional[str] = None
    for sp in preferred_splits:
        if sp in available_splits:
            chosen_split = sp
            break
    if chosen_split is None:
        chosen_split = available_splits[0]

    ds = _datasets_server_rows(dataset_name, chosen_config, chosen_split, max_rows=max_rows)
    return ds, chosen_split


def _download_file(url: str, out_path: Path, timeout_s: float = 120.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url=url)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def _list_parquet_urls_via_hf_api(dataset_name: str, dataset_config: str, split: str) -> list[str]:
    base = os.getenv("HF_HUB_ENDPOINT", "https://huggingface.co").rstrip("/")
    safe_dataset = dataset_name.replace("/", "%2F")
    safe_config = urllib.parse.quote(dataset_config, safe="")
    safe_split = urllib.parse.quote(split, safe="")

    urls: list[str] = []
    for i in range(0, 2048):
        url = f"{base}/api/datasets/{safe_dataset}/parquet/{safe_config}/{safe_split}/{i}.parquet"
        try:
            req = urllib.request.Request(url=url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30.0):
                pass
            urls.append(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            raise
    return urls


def _load_via_datasets_server_parquet(
    dataset_name: str,
    dataset_config: Optional[str],
    preferred_splits: list[str],
    cache_dir: Optional[str],
):
    base = os.getenv("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co").rstrip("/")
    ds_param = urllib.parse.quote(dataset_name, safe="")
    q = f"{base}/parquet?dataset={ds_param}"
    meta: dict[str, Any] = {}
    try:
        if dataset_config:
            cfg_param = urllib.parse.quote(dataset_config, safe="")
            q_cfg = f"{q}&config={cfg_param}"
            meta = _http_get_json(q_cfg)
        else:
            meta = _http_get_json(q)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    parquet_files = meta.get("parquet_files")
    if not isinstance(parquet_files, list) or not parquet_files:
        parquet_files = []

    split_to_urls: dict[str, list[str]] = {}
    for item in parquet_files:
        if not isinstance(item, dict):
            continue
        split = item.get("split")
        url = item.get("url")
        cfg = item.get("config")
        if dataset_config and cfg and str(cfg) != dataset_config:
            continue
        if isinstance(split, str) and isinstance(url, str) and url:
            split_to_urls.setdefault(split, []).append(url)

    if dataset_config and not split_to_urls:
        for s in preferred_splits:
            urls = _list_parquet_urls_via_hf_api(dataset_name, dataset_config, s)
            if urls:
                split_to_urls[s] = urls
    if not split_to_urls:
        raise RuntimeError(f"No parquet urls found for {dataset_name}/{dataset_config}")

    chosen_split: Optional[str] = None
    for s in preferred_splits:
        if s in split_to_urls:
            chosen_split = s
            break
    if chosen_split is None and split_to_urls:
        chosen_split = sorted(split_to_urls.keys())[0]
    if chosen_split is None:
        raise RuntimeError(f"No parquet splits found for {dataset_name}/{dataset_config}")
    if not split_to_urls.get(chosen_split):
        raise RuntimeError(f"No parquet urls found for {dataset_name}/{dataset_config}:{chosen_split}")

    local_dir = (Path(cache_dir) / "parquet_cache") if cache_dir else (Path("/root/workspace/train") / ".parquet_cache")
    safe = dataset_name.replace("/", "__")
    if dataset_config:
        safe += f"__{dataset_config}"
    local_dir = local_dir / safe / chosen_split
    local_paths: list[str] = []
    for i, url in enumerate(split_to_urls[chosen_split]):
        out_path = local_dir / f"{i}.parquet"
        if not out_path.exists():
            _download_file(url, out_path)
        local_paths.append(str(out_path))

    ds = load_dataset("parquet", data_files={chosen_split: local_paths}, split=chosen_split)
    return ds, chosen_split


def _download_medmcqa(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "medmcqa_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    ds, _ = _load_preferred_split(
        "medmcqa",
        None,
        ["test", "validation", "train"],
        cache_dir,
        max_rows=_max_rows_hint(max_samples),
    )
    rows: list[dict[str, Any]] = []
    for ex in ds:
        choices = [_as_str(ex.get("opa")), _as_str(ex.get("opb")), _as_str(ex.get("opc")), _as_str(ex.get("opd"))]
        if not any(choices):
            raw_choices = ex.get("choices", ex.get("options"))
            if isinstance(raw_choices, list) and raw_choices:
                choices = [_as_str(x) for x in raw_choices[:4]]
        if len(choices) < 4 or not all(choices[:4]):
            continue

        cop = ex.get("cop", ex.get("answer", ex.get("label")))
        if isinstance(cop, str):
            s = cop.strip()
            if s.isdigit():
                cop = int(s)
            elif s.upper() in ["A", "B", "C", "D"]:
                cop = "ABCD".index(s.upper())
        if isinstance(cop, int) and cop in [1, 2, 3, 4]:
            cop = cop - 1
        if not isinstance(cop, int) or cop < 0 or cop > 3:
            continue

        rows.append(
            {
                "id": _as_str(ex.get("id", ex.get("qid", ex.get("question_id")))),
                "question": _as_str(ex.get("question")),
                "choices": choices[:4],
                "answer": _letter_from_index(cop),
                "subject": _as_str(ex.get("subject")),
            }
        )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    _write_jsonl(out_path, rows)
    return out_path


def _extract_medqa_options(ex: dict[str, Any]) -> Optional[list[str]]:
    for key in ["options", "choices", "answer_options"]:
        val = ex.get(key)
        if isinstance(val, list) and val:
            return [_as_str(x) for x in val[:4]]
        if isinstance(val, dict):
            if all(k in val for k in ["A", "B", "C", "D"]):
                return [_as_str(val["A"]), _as_str(val["B"]), _as_str(val["C"]), _as_str(val["D"])]
    if all(isinstance(ex.get(k), str) for k in ["A", "B", "C", "D"]):
        return [_as_str(ex["A"]), _as_str(ex["B"]), _as_str(ex["C"]), _as_str(ex["D"])]
    return None


def _extract_medqa_answer_letter(ex: dict[str, Any], options: list[str]) -> Optional[str]:
    ans = ex.get("answer", ex.get("label", ex.get("gold", ex.get("target"))))
    if isinstance(ans, int) and 0 <= ans < min(4, len(options)):
        return _letter_from_index(ans)
    if isinstance(ans, int) and ans in [1, 2, 3, 4]:
        return _letter_from_index(ans - 1)
    if isinstance(ans, str) and ans.strip():
        s = ans.strip()
        if s.isdigit() and int(s) in [1, 2, 3, 4]:
            return _letter_from_index(int(s) - 1)
        if s.upper() in ["A", "B", "C", "D"]:
            return s.upper()
        if s in options:
            idx = options.index(s)
            if 0 <= idx < 4:
                return _letter_from_index(idx)
    return None


def _download_medqa_usmle(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "medqa_usmle_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    ds, _ = _load_preferred_split(
        "GBaker/MedQA-USMLE-4-options",
        None,
        ["test", "validation", "train"],
        cache_dir,
        max_rows=_max_rows_hint(max_samples),
    )
    rows: list[dict[str, Any]] = []
    for ex in ds:
        question = _as_str(ex.get("question", ex.get("prompt", ex.get("input"))))
        options = _extract_medqa_options(ex)
        if not question or not options or len(options) < 4:
            continue
        gold = _extract_medqa_answer_letter(ex, options)
        if gold is None:
            continue
        rows.append(
            {
                "id": _as_str(ex.get("id", ex.get("qid", ex.get("question_id")))),
                "question": question,
                "choices": options[:4],
                "answer": gold,
                "subject": "medqa_usmle",
            }
        )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    _write_jsonl(out_path, rows)
    return out_path


def _download_pubmedqa(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "pubmedqa_pqa_labeled_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    ds, _ = _load_preferred_split(
        "pubmed_qa",
        "pqa_labeled",
        ["test", "validation", "train"],
        cache_dir,
        max_rows=_max_rows_hint(max_samples),
    )
    rows: list[dict[str, Any]] = []
    for ex in ds:
        question = _as_str(ex.get("question", ex.get("qry", ex.get("prompt"))))
        label = _as_str(ex.get("final_decision", ex.get("answer", ex.get("label")))).lower()
        if label not in ["yes", "no", "maybe"]:
            continue
        ctx = ex.get("context")
        context = ""
        if isinstance(ctx, dict):
            contexts = ctx.get("contexts")
            if isinstance(contexts, list):
                context = "\n".join([_as_str(x) for x in contexts if _as_str(x)])
        if not context:
            context = _as_str(ex.get("context", ex.get("abstract", ex.get("passage", ""))))
        rows.append(
            {
                "id": _as_str(ex.get("pubid", ex.get("id", ex.get("qid")))),
                "question": question,
                "context": context,
                "final_decision": label,
            }
        )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    _write_jsonl(out_path, rows)
    return out_path


def _extract_cmexam_options(ex: dict[str, Any]) -> tuple[list[str], Optional[str]]:
    raw = ex.get("Options", ex.get("options"))
    if not isinstance(raw, list) or not raw:
        return [], None
    kv: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict):
            k = item.get("key", item.get("label"))
            v = item.get("value", item.get("text"))
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                kv[k.strip().upper()] = v.strip()
    if not kv:
        return [], None
    keys = [k for k in ["A", "B", "C", "D", "E"] if k in kv]
    if len(keys) < 2:
        keys = sorted(kv.keys())
    choices = [kv[k] for k in keys]
    return choices, None


def _download_cmexam(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "cmexam_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    ds, _ = _load_preferred_split(
        "fzkuji/CMExam",
        None,
        ["test", "validation", "dev", "train"],
        cache_dir,
        max_rows=_max_rows_hint(max_samples),
    )
    rows: list[dict[str, Any]] = []
    for ex in ds:
        question = _as_str(ex.get("Question", ex.get("question", ex.get("prompt", ex.get("input")))))
        choices, _ = _extract_cmexam_options(ex)
        if not question or len(choices) < 2:
            continue
        ans = _as_str(ex.get("Answer", ex.get("answer", ex.get("label")))).strip().upper()
        if len(ans) != 1 or not ("A" <= ans <= "Z"):
            continue
        idx = ord(ans) - ord("A")
        if idx < 0 or idx >= len(choices):
            continue
        rows.append(
            {
                "id": _as_str(ex.get("id", ex.get("qid", ex.get("question_id")))),
                "question": question,
                "choices": choices,
                "answer": ans,
                "subject": "cmexam",
            }
        )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    if not rows:
        raise RuntimeError("No examples loaded for CMExam.")
    _write_jsonl(out_path, rows)
    return out_path


def _download_mmedbench_zh(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "mmedbench_zh_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path

    base = os.getenv("HF_HUB_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{base}/datasets/Henrychur/MMedBench/resolve/main/MMedBench.zip"
    store_dir = (Path(cache_dir) / "mmedbench_assets") if cache_dir else (Path("/root/workspace/train") / ".mmedbench_assets")
    zip_path = store_dir / "MMedBench.zip"
    if not zip_path.exists():
        _download_file(url, zip_path)

    member: Optional[str] = None
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            low = name.lower()
            if low.endswith("/test/chinese.jsonl") or low.endswith("test/chinese.jsonl"):
                member = name
                break
        if member is None:
            raise RuntimeError("Cannot find Chinese test split in MMedBench.zip")

        rows: list[dict[str, Any]] = []
        with zf.open(member) as f:
            for raw in f:
                if max_samples is not None and max_samples > 0 and len(rows) >= max_samples:
                    break
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                ex = json.loads(line)
                question = _as_str(ex.get("question"))
                options = ex.get("options")
                if not question or not isinstance(options, dict) or not options:
                    continue
                ordered_keys = [k for k in ["A", "B", "C", "D", "E"] if k in options]
                if len(ordered_keys) < 2:
                    continue
                choices = [_as_str(options[k]) for k in ordered_keys]
                ans = _as_str(ex.get("answer_idx", ex.get("answer", ex.get("label")))).strip().upper()
                if len(ans) != 1 or ans not in ordered_keys:
                    continue
                rows.append(
                    {
                        "id": _as_str(ex.get("id")),
                        "question": question,
                        "choices": choices,
                        "answer": ans,
                        "subject": "mmedbench_zh",
                    }
                )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if not rows:
        raise RuntimeError("No examples loaded for MMedBench Chinese.")
    _write_jsonl(out_path, rows)
    return out_path


def _download_cmb(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "cmb_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    ds, _ = _load_preferred_split(
        "FreedomIntelligence/CMB",
        None,
        ["test", "validation", "dev", "train"],
        cache_dir,
        max_rows=_max_rows_hint(max_samples),
    )
    rows: list[dict[str, Any]] = []
    for ex in ds:
        question = _as_str(ex.get("question", ex.get("prompt", ex.get("input", ex.get("Question")))))
        options = _extract_medqa_options(ex)
        if not options or len(options) < 4:
            if all(_as_str(ex.get(k)) for k in ["A", "B", "C", "D"]):
                options = [_as_str(ex["A"]), _as_str(ex["B"]), _as_str(ex["C"]), _as_str(ex["D"])]
        if not question or not options or len(options) < 4:
            continue
        gold = _extract_medqa_answer_letter(ex, options)
        if gold is None:
            gold = _as_str(ex.get("Answer", ex.get("answer"))).strip().upper()
            if gold not in ["A", "B", "C", "D"]:
                continue
        rows.append(
            {
                "id": _as_str(ex.get("id", ex.get("qid", ex.get("question_id")))),
                "question": question,
                "choices": options[:4],
                "answer": gold,
                "subject": _as_str(ex.get("subject", "cmb")),
            }
        )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    _write_jsonl(out_path, rows)
    return out_path


_CMMLU_MED_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "college_medical_statistics",
]


def _download_cmmlu_medical(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "cmmlu_medical_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    rows: list[dict[str, Any]] = []
    for subject in _CMMLU_MED_SUBJECTS:
        try:
            ds, _ = _load_preferred_split(
                "haonan-li/cmmlu",
                subject,
                ["test", "validation", "dev", "train"],
                cache_dir,
                max_rows=_max_rows_hint(max_samples),
            )
        except Exception:
            continue
        for ex in ds:
            question = _as_str(ex.get("question", ex.get("prompt", ex.get("input"))))
            options: Optional[list[str]] = None
            if all(_as_str(ex.get(k)) for k in ["A", "B", "C", "D"]):
                options = [_as_str(ex["A"]), _as_str(ex["B"]), _as_str(ex["C"]), _as_str(ex["D"])]
            else:
                options = _extract_medqa_options(ex)
            if not question or not options or len(options) < 4:
                continue
            gold = _extract_medqa_answer_letter(ex, options)
            if gold is None:
                continue
            rows.append(
                {
                    "id": _as_str(ex.get("id", ex.get("qid", ex.get("question_id")))),
                    "question": question,
                    "choices": options[:4],
                    "answer": gold,
                    "subject": subject,
                }
            )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    if not rows:
        raise RuntimeError("No examples loaded for CMMLU medical.")
    _write_jsonl(out_path, rows)
    return out_path


_MMLU_MED_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
]


def _download_mmlu_medical(out_dir: Path, cache_dir: Optional[str], max_samples: Optional[int], seed: int) -> Path:
    out_path = out_dir / "mmlu_medical_test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0 and (max_samples is None or max_samples <= 0):
        return out_path
    rows: list[dict[str, Any]] = []
    for subject in _MMLU_MED_SUBJECTS:
        ds, _ = _load_preferred_split(
            "cais/mmlu",
            subject,
            ["test", "validation", "dev", "train"],
            cache_dir,
            max_rows=_max_rows_hint(max_samples),
        )
        for ex in ds:
            question = _as_str(ex.get("question"))
            choices = ex.get("choices")
            answer = ex.get("answer")
            if not question or not isinstance(choices, list) or len(choices) < 4 or not isinstance(answer, int):
                continue
            if answer < 0 or answer > 3:
                continue
            rows.append(
                {
                    "id": _as_str(ex.get("id", ex.get("qid"))),
                    "question": question,
                    "choices": [_as_str(c) for c in choices[:4]],
                    "answer": _letter_from_index(answer),
                    "subject": subject,
                }
            )
    rows = _maybe_limit(rows, max_samples=max_samples, seed=seed)
    _write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/root/workspace/train/benchmark_data")
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        default="all",
        help="all|medqa|medmcqa|pubmedqa|mmlu|cmb|cmmlu_med|cmexam|mmedbench_zh",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir else None

    outputs: dict[str, str] = {}
    def _run_one(key: str, fn) -> None:
        try:
            outputs[key] = str(fn(out_dir, cache_dir, args.max_samples, args.seed))
        except Exception as e:
            outputs[f"{key}_error"] = str(e)
            if args.only != "all":
                raise

    if args.only in ["all", "medqa"]:
        _run_one("medqa", _download_medqa_usmle)
    if args.only in ["all", "medmcqa"]:
        _run_one("medmcqa", _download_medmcqa)
    if args.only in ["all", "pubmedqa"]:
        _run_one("pubmedqa", _download_pubmedqa)
    if args.only in ["all", "cmb"]:
        _run_one("cmb", _download_cmb)
    if args.only in ["all", "cmexam"]:
        _run_one("cmexam", _download_cmexam)
    if args.only in ["all", "mmedbench_zh"]:
        _run_one("mmedbench_zh", _download_mmedbench_zh)
    if args.only in ["all", "cmmlu_med"]:
        _run_one("cmmlu_medical", _download_cmmlu_medical)
    if args.only in ["all", "mmlu"]:
        _run_one("mmlu_medical", _download_mmlu_medical)

    print(json.dumps({"out_dir": str(out_dir), "files": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
