import asyncio
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import httpx
import jiwer
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from ocr_evaluate.ocr_gemini import client  # Reuse initialized Gemini client and env
from google.genai.types import GenerateContentConfig, Part

# --- Configuration (defaults; can be overridden by CLI later if extended) ---
MAX_CONCURRENT_REQUESTS = 20
RETRY_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2
BACKOFF_MAX_SECONDS = 20
TOTAL_EVALUATION_RUNS = 10


MODELS_TO_TEST = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",    
]

# Resolve benchmark directory relative to this script's folder
PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# Output CSVs for this pipeline
SUMMARY_REPORT_CSV = "evaluation_summary_report_fewshot.csv"
DETAILED_OUTPUTS_CSV = "detailed_inference_outputs_fewshot.csv"

# Few-shot parameters
FEWSHOT_K = 3
FEWSHOT_SEED = 42

# Prompt instruction
SYSTEM_INSTRUCTION = (
    "You are an absolute expert on Tibetan. Perform OCR on the provided page images. "
    "Return only clean Tibetan text, preserving natural line breaks. Do not use markdown."
)
USER_PROMPT_EXEMPLAR = "Here is a similar Pecha page. Please OCR it."
USER_PROMPT_TARGET = "Here is the target page. Please OCR it."


@dataclass
class Exemplar:
    image_bytes: bytes
    mime_type: str
    ground_truth_text: str


def read_existing_summary_keys(csv_path: Path) -> Set[Tuple[str, str, str]]:
    keys: Set[Tuple[str, str, str]] = set()
    if not csv_path.exists():
        return keys
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("model_name")
                dataset = row.get("dataset")
                filename = row.get("filename")
                if model_name and dataset and filename:
                    keys.add((model_name, dataset, filename))
    except Exception:
        return set()
    return keys


def list_images_with_txt(dataset_path: Path) -> List[Path]:
    images = sorted(list(dataset_path.glob("*.png"))) + sorted(list(dataset_path.glob("*.jpg")))
    return [p for p in images if p.with_suffix(".txt").exists()]


def guess_mime(path: Path) -> str:
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".png":
        return "image/png"
    if s == ".webp":
        return "image/webp"
    return "application/octet-stream"


def sample_exemplars(dataset_path: Path, exclude_filename: str, k: int, seed: int) -> List[Exemplar]:
    candidates = [p for p in list_images_with_txt(dataset_path) if p.name != exclude_filename]
    if not candidates:
        return []
    rng = random.Random(seed)
    sample = candidates if len(candidates) <= k else rng.sample(candidates, k)
    exemplars: List[Exemplar] = []
    for p in sample:
        img_bytes = p.read_bytes()
        mime = guess_mime(p)
        gt_text = p.with_suffix(".txt").read_text(encoding="utf-8").strip()
        exemplars.append(Exemplar(img_bytes, mime, gt_text))
    return exemplars


def build_fewshot_contents(exemplars: Sequence[Exemplar], target_bytes: bytes, target_mime: str) -> List[object]:
    # The contents array alternates text and image Parts as multi-turn context
    contents: List[object] = []
    contents.append(SYSTEM_INSTRUCTION)
    for ex in exemplars:
        contents.append(USER_PROMPT_EXEMPLAR)
        contents.append(Part.from_bytes(data=ex.image_bytes, mime_type=ex.mime_type))
        contents.append(ex.ground_truth_text)
    contents.append(USER_PROMPT_TARGET)
    contents.append(Part.from_bytes(data=target_bytes, mime_type=target_mime))
    return contents


async def generate_with_retries(contents: List[object], model_name: str) -> Optional[str]:
    config = GenerateContentConfig(max_output_tokens=4000)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception as e:
            # HTTP 429 backoff if available
            retry_after_secs = None
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                try:
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        retry_after_secs = int(float(ra))
                except Exception:
                    pass
            base_delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
            jitter = random.uniform(0.0, 0.5)
            delay = max(base_delay + jitter, retry_after_secs or 0)
            tqdm.write(f"[fewshot] Error attempt {attempt+1}/{RETRY_ATTEMPTS} on model {model_name}: {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    return None


async def process_single_image(image_path: Path, model_name: str, fewshot_k: int, fewshot_seed: int) -> Optional[dict]:
    dataset_path = image_path.parent
    gt_path = image_path.with_suffix(".txt")
    ground_truth = gt_path.read_text(encoding="utf-8").strip()

    # Prepare exemplars once per target for all runs
    exemplars = sample_exemplars(dataset_path, exclude_filename=image_path.name, k=fewshot_k, seed=fewshot_seed)

    target_bytes = image_path.read_bytes()
    target_mime = guess_mime(image_path)

    # For now, reuse same exemplars each run to reduce variance from sampling
    cer_scores: List[float] = []
    detailed_first_run: Optional[dict] = None

    for run_idx in range(TOTAL_EVALUATION_RUNS):
        contents = build_fewshot_contents(exemplars, target_bytes, target_mime)
        ocr_text = await generate_with_retries(contents, model_name)
        if ocr_text is None:
            tqdm.write(f"Failed to OCR {image_path.name} with {model_name} after retries.")
            continue
        cer = jiwer.cer(ground_truth, ocr_text)
        cer_scores.append(cer)
        if run_idx == 0:
            detailed_first_run = {
                "model_name": model_name,
                "dataset": dataset_path.name,
                "filename": image_path.name,
                "ground_truth": ground_truth,
                "ocr_output": ocr_text,
                "cer": cer,
            }

    if not cer_scores:
        return None

    return {
        "filename": image_path.name,
        "cer_scores": cer_scores,
        "detailed_first_run": detailed_first_run,
    }


async def run_evaluation_for_model_and_dataset(model_name: str, dataset_path: Path, pbar: tqdm, cer_run_columns: List[str], existing_keys: Set[Tuple[str, str, str]], fewshot_k: int, fewshot_seed: int):
    images = list_images_with_txt(dataset_path)

    # Filter already completed by checkpoint
    images = [p for p in images if (model_name, dataset_path.name, p.name) not in existing_keys]

    if not images:
        tqdm.write(f"Skipping {model_name} | {dataset_path.name}: already complete.")
        return [], []

    # Concurrency semaphore
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def task_wrapper(path: Path):
        async with sem:
            return await process_single_image(path, model_name, fewshot_k, fewshot_seed)

    results = await tqdm_asyncio.gather(*[task_wrapper(p) for p in images], leave=False, desc=f"Eval {model_name} | {dataset_path.name}")

    summary_rows: List[dict] = []
    detailed_rows: List[dict] = []

    for res in results:
        if not res:
            continue
        filename = res["filename"]
        cer_scores: List[float] = res["cer_scores"]
        mean_cer = float(np.mean(cer_scores))
        std_dev_cer = float(np.std(cer_scores)) if len(cer_scores) > 1 else 0.0
        row = {
            "model_name": model_name,
            "dataset": dataset_path.name,
            "filename": filename,
            "mean_cer": mean_cer,
            "std_dev_cer": std_dev_cer,
        }
        for idx, col in enumerate(cer_run_columns):
            row[col] = cer_scores[idx] if idx < len(cer_scores) else None
        summary_rows.append(row)
        if res["detailed_first_run"]:
            detailed_rows.append(res["detailed_first_run"])

    return summary_rows, detailed_rows


async def main():
    datasets = [d for d in BENCHMARK_DIR.iterdir() if d.is_dir()]

    total_tasks = len(MODELS_TO_TEST) * len(datasets)
    overall_pbar = tqdm(total=total_tasks, desc="Few-shot Overall Progress")

    cer_run_columns = [f"cer_run_{i+1}" for i in range(TOTAL_EVALUATION_RUNS)]
    summary_fieldnames = ["model_name", "dataset", "filename", *cer_run_columns, "mean_cer", "std_dev_cer"]

    summary_path = Path(SUMMARY_REPORT_CSV)
    detailed_path = Path(DETAILED_OUTPUTS_CSV)

    summary_exists = summary_path.exists()
    detailed_exists = detailed_path.exists()

    existing_keys = read_existing_summary_keys(summary_path)

    with open(summary_path, "a" if summary_exists else "w", newline="", encoding="utf-8") as f_summary, \
         open(detailed_path, "a" if detailed_exists else "w", newline="", encoding="utf-8") as f_detailed:

        summary_writer = csv.DictWriter(f_summary, fieldnames=summary_fieldnames)
        if not summary_exists:
            summary_writer.writeheader()

        detailed_writer = csv.DictWriter(f_detailed, fieldnames=["model_name", "dataset", "filename", "ground_truth", "ocr_output", "cer"])
        if not detailed_exists:
            detailed_writer.writeheader()

        for model in MODELS_TO_TEST:
            for dataset_path in datasets:
                summary, details = await run_evaluation_for_model_and_dataset(
                    model, dataset_path, overall_pbar, cer_run_columns, existing_keys, FEWSHOT_K, FEWSHOT_SEED
                )
                if summary:
                    summary_writer.writerows(summary)
                    f_summary.flush()
                    for row in summary:
                        existing_keys.add((row["model_name"], row["dataset"], row["filename"]))
                if details:
                    detailed_writer.writerows(details)
                    f_detailed.flush()
                overall_pbar.update(1)

    overall_pbar.close()
    print(f"\nFew-shot evaluation complete. Reports saved to '{SUMMARY_REPORT_CSV}' and '{DETAILED_OUTPUTS_CSV}'.")


if __name__ == "__main__":
    asyncio.run(main()) 