import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: Path
    fmt: str


@dataclass(frozen=True)
class Example:
    example_id: str
    prompt_messages: list[dict[str, str]]
    gold: str
    meta: dict[str, Any]


def _parse_tasks(task_args: list[str]) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for raw in task_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --task: {raw}. Expected name=PATH:FORMAT")
        name, rest = raw.split("=", 1)
        if ":" not in rest:
            raise ValueError(f"Invalid --task: {raw}. Expected name=PATH:FORMAT")
        path_str, fmt = rest.rsplit(":", 1)
        fmt = fmt.strip().lower()
        path = Path(path_str).expanduser()
        tasks.append(TaskSpec(name=name.strip(), path=path, fmt=fmt))
    if not tasks:
        raise ValueError("At least one --task is required.")
    return tasks


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e


def _normalize_choice_text(choice: Any) -> str:
    if isinstance(choice, str):
        return choice.strip()
    if isinstance(choice, dict):
        for key in ["text", "content", "value", "label"]:
            if key in choice and isinstance(choice[key], str):
                return choice[key].strip()
    return str(choice).strip()


def _extract_mcq_fields(obj: dict[str, Any]) -> tuple[str, list[str], str]:
    question = None
    for key in ["question", "prompt", "input", "query", "stem"]:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            question = val.strip()
            break
    if question is None:
        raise ValueError("Missing question field.")

    choices = None
    if isinstance(obj.get("choices"), list) and obj["choices"]:
        choices = [_normalize_choice_text(c) for c in obj["choices"]]
    elif all(isinstance(obj.get(k), str) for k in ["A", "B", "C", "D"]):
        choices = [obj["A"], obj["B"], obj["C"], obj["D"]]
    elif all(isinstance(obj.get(k), str) for k in ["opa", "opb", "opc", "opd"]):
        choices = [obj["opa"], obj["opb"], obj["opc"], obj["opd"]]

    if not choices or len(choices) < 2:
        raise ValueError("Missing choices field.")

    answer = obj.get("answer", obj.get("label", obj.get("gold", obj.get("target"))))
    if isinstance(answer, int):
        idx = answer
        if 1 <= idx <= len(choices):
            idx = idx - 1
        if idx < 0 or idx >= len(choices):
            raise ValueError("Answer index out of range.")
        if answer >= 26:
            raise ValueError("Too many choices.")
        return question, choices, chr(ord("A") + idx)
    if isinstance(answer, str) and answer.strip():
        ans = answer.strip()
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(choices):
                idx = idx - 1
            if 0 <= idx < len(choices):
                if idx >= 26:
                    raise ValueError("Too many choices.")
                return question, choices, chr(ord("A") + idx)
            raise ValueError("Answer index out of range.")
        if len(ans) == 1 and "A" <= ans.upper() <= "Z":
            idx = ord(ans.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return question, choices, ans.upper()
            raise ValueError("Answer letter out of range.")
        if ans in choices:
            idx = choices.index(ans)
            return question, choices, chr(ord("A") + idx)
    raise ValueError("Missing or invalid answer field.")


def _detect_lang(text: str) -> str:
    if not text:
        return "zh"
    cjk = 0
    latin = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
    if cjk >= max(8, latin * 2):
        return "zh"
    if latin >= max(16, cjk * 2):
        return "en"
    return "zh"


def _augment_system(system: str, *, lang: str, thinking: str) -> str:
    system = system.strip()
    if thinking == "off":
        if lang == "zh":
            extra = "不要输出推理过程/思考过程/解释，只输出最终答案。"
        else:
            extra = "Do not output reasoning. Output only the final answer."
        return (system + "\n\n" + extra).strip() if system else extra
    return system


def _build_mcq_messages(
    question: str,
    choices: list[str],
    system: str,
    *,
    lang: str,
    thinking: str,
) -> list[dict[str, str]]:
    lettered = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)])
    allowed = "/".join([chr(ord("A") + i) for i in range(len(choices))])
    if lang == "en":
        user = (
            f"{question}\n\n{lettered}\n\n"
            f"Output only the final answer letter ({allowed}). Output a single letter only."
        )
    else:
        user = f"{question}\n\n{lettered}\n\n只输出最终答案的选项字母（{allowed}），不要输出其它内容。"
    return [
        {"role": "system", "content": _augment_system(system, lang=lang, thinking=thinking)},
        {"role": "user", "content": user},
    ]


def _build_pubmedqa_messages(
    question: str,
    context: str,
    system: str,
    *,
    lang: str,
    thinking: str,
) -> list[dict[str, str]]:
    if lang == "en":
        user = (
            f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Output only the final answer (yes/no/maybe). Output a single word only."
        )
    else:
        user = f"Context:\n{context}\n\nQuestion:\n{question}\n\n只输出最终答案（yes/no/maybe），不要输出其它内容。"
    return [
        {"role": "system", "content": _augment_system(system, lang=lang, thinking=thinking)},
        {"role": "user", "content": user},
    ]


def _extract_pubmedqa_fields(obj: dict[str, Any]) -> tuple[str, str, str]:
    question = obj.get("question", obj.get("qry", obj.get("prompt")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing question field.")
    context = obj.get("context", obj.get("contexts", obj.get("passage", obj.get("abstract", ""))))
    if isinstance(context, list):
        context = "\n".join([str(x) for x in context])
    if not isinstance(context, str):
        context = str(context)
    gold = obj.get("final_decision", obj.get("answer", obj.get("label", obj.get("gold"))))
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("Missing label field.")
    gold_norm = gold.strip().lower()
    if gold_norm not in ["yes", "no", "maybe"]:
        raise ValueError(f"Invalid label: {gold}")
    return question.strip(), context.strip(), gold_norm


def _load_examples(
    task: TaskSpec,
    system: str,
    max_samples: Optional[int],
    *,
    lang: str,
    thinking: str,
) -> list[Example]:
    if not task.path.exists():
        raise FileNotFoundError(f"Dataset not found: {task.path}")
    examples: list[Example] = []
    for idx, obj in enumerate(_iter_jsonl(task.path)):
        if max_samples is not None and len(examples) >= max_samples:
            break
        example_id = str(obj.get("id", obj.get("qid", obj.get("question_id", idx))))
        meta: dict[str, Any] = {}
        if task.fmt in ["mcq-jsonl", "medqa-jsonl", "medmcqa-jsonl", "mmlu-jsonl"]:
            question, choices, gold = _extract_mcq_fields(obj)
            meta["choices"] = choices
            meta["question"] = question
            if "subject" in obj:
                meta["subject"] = obj["subject"]
            ex_lang = _detect_lang(question) if lang == "auto" else lang
            messages = _build_mcq_messages(question, choices, system=system, lang=ex_lang, thinking=thinking)
        elif task.fmt in ["pubmedqa-jsonl", "ynm-jsonl"]:
            question, context, gold = _extract_pubmedqa_fields(obj)
            meta["question"] = question
            meta["context"] = context
            ex_lang = _detect_lang(question) if lang == "auto" else lang
            messages = _build_pubmedqa_messages(question, context, system=system, lang=ex_lang, thinking=thinking)
        else:
            raise ValueError(f"Unknown task format: {task.fmt}")
        examples.append(Example(example_id=example_id, prompt_messages=messages, gold=gold, meta=meta))
    return examples


_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_FULLWIDTH_TRANS = str.maketrans(
    {"Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e"}
)

_MCQ_LETTER_RE = re.compile(r"(?i)(?:^|\b)(answer|final|答案)\s*[:：]?\s*([ABCDE])\b")
_MCQ_HINT_RE = re.compile(r"(?i)(?:选|选择|答案|选项)\s*[:：]?\s*([ABCDE])")
_MCQ_STANDALONE_RE = re.compile(r"(?i)(?<![0-9A-Z_])([ABCDE])(?![0-9A-Z_])")
_MCQ_FIRST_TOKEN_RE = re.compile(r"(?is)^\s*(?:final|answer|答案)?\s*[:：]?\s*[\(\[]?\s*([ABCDE])\s*[\)\].:：、]?\s*$")
_MCQ_LEADING_RE = re.compile(r"(?is)^\s*[\(\[]?\s*([ABCDE])\s*[\)\].:：、]?\b")
_YNM_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_YNM_ZH_YES_RE = re.compile(r"(?:\b|^)(是|对|正确|可以)(?:\b|$)")
_YNM_ZH_NO_RE = re.compile(r"(?:\b|^)(否|不|错误|不能)(?:\b|$)")
_YNM_ZH_MAYBE_RE = re.compile(r"(?:\b|^)(可能|不确定|未知)(?:\b|$)")


def _parse_mcq_answer(text: str) -> Optional[str]:
    text = _THINK_BLOCK_RE.sub(" ", text).translate(_FULLWIDTH_TRANS)
    text = text.strip()
    m0 = _MCQ_FIRST_TOKEN_RE.search(text)
    if m0:
        return m0.group(1).upper()
    m00 = _MCQ_LEADING_RE.search(text)
    if m00:
        return m00.group(1).upper()
    m = _MCQ_LETTER_RE.search(text)
    if m:
        return m.group(2).upper()
    m2 = _MCQ_HINT_RE.search(text)
    if m2:
        return m2.group(1).upper()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ["answer", "final", "choice", "label"]:
                v = obj.get(key)
                if isinstance(v, str) and v.strip().upper() in ["A", "B", "C", "D", "E"]:
                    return v.strip().upper()
    all_letters = _MCQ_STANDALONE_RE.findall(text)
    if all_letters:
        return all_letters[-1].upper()
    return None


def _parse_ynm_answer(text: str) -> Optional[str]:
    text = _THINK_BLOCK_RE.sub(" ", text).translate(_FULLWIDTH_TRANS)
    text = text.strip()
    matches = _YNM_RE.findall(text)
    if matches:
        return matches[-1].lower()
    if _YNM_ZH_MAYBE_RE.search(text):
        return "maybe"
    if _YNM_ZH_YES_RE.search(text):
        return "yes"
    if _YNM_ZH_NO_RE.search(text):
        return "no"
    return None


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _chat_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[list[str]],
    chat_template_kwargs: Optional[dict[str, Any]],
    timeout_s: float,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    base = endpoint.rstrip("/")
    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            raw = _post_json(url, payload, headers=headers, timeout_s=timeout_s)
            content = raw["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = str(content)
            return content, raw
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            last_error = e
            if attempt >= max_retries:
                break
            time.sleep(min(2.0**attempt, 8.0))
    raise RuntimeError(f"Request failed after retries: {last_error}") from last_error


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return cleaned.strip("_") or "model"


def _run_task(
    task: TaskSpec,
    examples: list[Example],
    endpoint: str,
    api_key: str,
    model: str,
    out_dir: Path,
    max_tokens: int,
    stop: Optional[list[str]],
    chat_template_kwargs: Optional[dict[str, Any]],
    timeout_s: float,
    max_retries: int,
    retry_invalid: int,
) -> dict[str, Any]:
    out_path = out_dir / f"{task.name}__{_safe_name(model)}.jsonl"
    correct = 0
    invalid = 0
    total = 0
    latencies_ms: list[float] = []

    def _allowed_mcq_letters(choices: list[str]) -> set[str]:
        return {chr(ord("A") + i) for i in range(min(len(choices), 26))}

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            total += 1
            t0 = time.monotonic()
            text_first, raw_first = _chat_completion(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                messages=ex.prompt_messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                stop=stop,
                chat_template_kwargs=chat_template_kwargs,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            dt = (time.monotonic() - t0) * 1000.0
            latencies_ms.append(dt)

            if task.fmt in ["pubmedqa-jsonl", "ynm-jsonl"]:
                pred_first = _parse_ynm_answer(text_first)
                pred = pred_first
                raw = raw_first
                text = text_first
                attempts = 1
                if (pred is None) and retry_invalid > 0:
                    followup = (
                        "只输出最终答案（yes/no/maybe），不要输出其它内容。"
                        if _detect_lang(ex.meta.get("question", "")) != "en"
                        else "Output only the final answer (yes/no/maybe). Output a single word only."
                    )
                    reprompt_messages = ex.prompt_messages + [
                        {"role": "assistant", "content": text_first},
                        {"role": "user", "content": followup},
                    ]
                    text, raw = _chat_completion(
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model,
                        messages=reprompt_messages,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=min(max_tokens, 16),
                        stop=stop,
                        chat_template_kwargs=chat_template_kwargs,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                    )
                    attempts += 1
                    pred = _parse_ynm_answer(text)
            else:
                allowed = _allowed_mcq_letters(ex.meta.get("choices", []))
                pred_first = _parse_mcq_answer(text_first)
                pred = pred_first if pred_first in allowed else None
                raw = raw_first
                text = text_first
                attempts = 1
                if pred is None and retry_invalid > 0:
                    allowed_str = "/".join(sorted(allowed)) if allowed else "A/B/C/D"
                    followup = (
                        f"只输出最终答案的选项字母（{allowed_str}），不要输出其它内容。"
                        if _detect_lang(ex.meta.get("question", "")) != "en"
                        else f"Output only the final answer letter ({allowed_str}). Output a single letter only."
                    )
                    reprompt_messages = ex.prompt_messages + [
                        {"role": "assistant", "content": text_first},
                        {"role": "user", "content": followup},
                    ]
                    text, raw = _chat_completion(
                        endpoint=endpoint,
                        api_key=api_key,
                        model=model,
                        messages=reprompt_messages,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=min(max_tokens, 16),
                        stop=stop,
                        chat_template_kwargs=chat_template_kwargs,
                        timeout_s=timeout_s,
                        max_retries=max_retries,
                    )
                    attempts += 1
                    pred2 = _parse_mcq_answer(text)
                    pred = pred2 if pred2 in allowed else None

            is_correct = pred == ex.gold if pred is not None else False
            if pred is None:
                invalid += 1
            if is_correct:
                correct += 1

            record = {
                "id": ex.example_id,
                "task": task.name,
                "model": model,
                "gold": ex.gold,
                "pred": pred,
                "correct": is_correct,
                "attempts": attempts,
                "latency_ms": dt,
                "output_text": text,
                "output_text_first": text_first,
                "meta": ex.meta,
                "raw": raw,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = correct / total if total else 0.0
    invalid_rate = invalid / total if total else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2] if latencies_ms else 0.0
    return {
        "task": task.name,
        "format": task.fmt,
        "model": model,
        "num_samples": total,
        "accuracy": acc,
        "invalid_rate": invalid_rate,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "predictions_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:1024")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--models", required=True, help="Comma-separated served model names.")
    parser.add_argument("--task", action="append", default=[], help="name=PATH:FORMAT")
    parser.add_argument("--system", default="You are a helpful medical assistant.")
    parser.add_argument("--lang", choices=["auto", "zh", "en"], default="auto")
    parser.add_argument("--thinking", choices=["off", "on"], default="off")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--retry-invalid", type=int, default=1)
    parser.add_argument("--no-stop-newline", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out-dir", default="benchmark_out")
    args = parser.parse_args()

    tasks = _parse_tasks(args.task)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("Empty --models")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"summary_{run_id}.json"

    all_results: list[dict[str, Any]] = []
    for task in tasks:
        examples = _load_examples(
            task,
            system=args.system,
            max_samples=args.max_samples,
            lang=args.lang,
            thinking=args.thinking,
        )
        if not examples:
            print(
                json.dumps(
                    {"task": task.name, "format": task.fmt, "skipped": True, "reason": "no_examples"},
                    ensure_ascii=False,
                )
            )
            continue
        for model in models:
            stop: Optional[list[str]]
            if args.no_stop_newline:
                stop = None
            elif args.thinking == "off":
                stop = ["\n"]
            else:
                stop = None

            chat_template_kwargs: Optional[dict[str, Any]] = None
            if args.thinking == "off":
                chat_template_kwargs = {"enable_thinking": False}
            res = _run_task(
                task=task,
                examples=examples,
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=model,
                out_dir=out_dir,
                max_tokens=args.max_tokens,
                stop=stop,
                chat_template_kwargs=chat_template_kwargs,
                timeout_s=args.timeout,
                max_retries=args.retries,
                retry_invalid=max(0, int(args.retry_invalid)),
            )
            all_results.append(res)
            print(json.dumps(res, ensure_ascii=False))

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"endpoint": args.endpoint, "results": all_results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
