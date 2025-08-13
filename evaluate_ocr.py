import asyncio
import csv
import numpy as np
from pathlib import Path
import jiwer
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from ocr_evaluate.ocr_gemini import do_ocr
import random
import httpx
from typing import Set, Tuple

# --- Configuration ---
MAX_CONCURRENT_REQUESTS = 15
RETRY_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 5
TOTAL_EVALUATION_RUNS = 10


BACKOFF_BASE_SECONDS = 2
BACKOFF_MAX_SECONDS = 20

MODELS_TO_TEST = [
    # "gemini-2.5-pro", # Temporarily skipped
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

BENCHMARK_DIR = "benchmark"

# --- CSV Output Files ---
SUMMARY_REPORT_CSV = "evaluation_summary_report.csv"
DETAILED_OUTPUTS_CSV = "detailed_inference_outputs.csv"


def read_existing_summary_keys(summary_csv_path: Path) -> Set[Tuple[str, str, str]]:
    keys: Set[Tuple[str, str, str]] = set()
    if not summary_csv_path.exists():
        return keys
    try:
        with open(summary_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("model_name")
                dataset = row.get("dataset")
                filename = row.get("filename")
                if model_name and dataset and filename:
                    keys.add((model_name, dataset, filename))
    except Exception:
        # If CSV is malformed/partial, ignore and treat as no checkpoints
        return set()
    return keys


async def process_single_image(image_path: Path, text_path: Path, model_name: str, semaphore: asyncio.Semaphore):
    """Processes a single image with retries and returns detailed results."""
    async with semaphore:
        ground_truth = text_path.read_text(encoding="utf-8").strip()
        for attempt in range(RETRY_ATTEMPTS):
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                suffix = image_path.suffix.lower()
                content_type = "image/png" if suffix == ".png" else "image/jpeg"
                ocr_text, _ = await do_ocr(image_bytes, image_path.name, content_type, model_name)

                cer = jiwer.cer(ground_truth, ocr_text)
                return {
                    "filename": image_path.name,
                    "ground_truth": ground_truth,
                    "ocr_output": ocr_text,
                    "cer": cer
                }
            except Exception as e:
                # Build rich error message
                err_msg = str(e) or e.__class__.__name__
                retry_after_secs = None
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    status = e.response.status_code
                    text_preview = (e.response.text or "").strip().replace("\n", " ")[:200]
                    err_msg = f"HTTP {status}: {text_preview}"
                    if status == 429:
                        try:
                            retry_after = e.response.headers.get("Retry-After")
                            if retry_after:
                                retry_after_secs = int(float(retry_after))
                        except Exception:
                            pass
                elif isinstance(e, httpx.RequestError):
                    err_msg = f"RequestError: {repr(e)}"

                if attempt >= RETRY_ATTEMPTS - 1:
                    tqdm.write(f"Failed to process {image_path.name} with {model_name} after {RETRY_ATTEMPTS} attempts. Error: {err_msg}")
                    return None
                # Exponential backoff with jitter
                base_delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
                jitter = random.uniform(0.0, 0.5)
                delay = base_delay + jitter
                if retry_after_secs is not None:
                    delay = max(delay, retry_after_secs)
                tqdm.write(f"Error on {image_path.name} with {model_name} (attempt {attempt+1}/{RETRY_ATTEMPTS}): {err_msg}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)


async def run_evaluation_for_model_and_dataset(model_name: str, dataset_path: Path, pbar: tqdm, cer_run_columns: list[str], existing_summary_keys: Set[Tuple[str, str, str]]):
    """Runs a full evaluation for a specific model and dataset, including multiple runs.
    Skips any (model, dataset, filename) already recorded in the existing summary CSV.
    """
    image_files = sorted(list(dataset_path.glob("*.png"))) + sorted(list(dataset_path.glob("*.jpg")))

    # Filter images that still need to be processed according to checkpoints
    images_to_process = [
        img for img in image_files
        if (model_name, dataset_path.name, img.name) not in existing_summary_keys and img.with_suffix(".txt").exists()
    ]

    if not images_to_process:
        tqdm.write(f"Skipping {model_name} | {dataset_path.name}: all images already present in summary CSV.")
        return [], []

    all_runs_results = {image.name: [] for image in images_to_process}
    detailed_outputs_for_run = []

    # Concurrency semaphore
    sem_general = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    for i in range(TOTAL_EVALUATION_RUNS):
        pbar.set_description(f"Model: {model_name} | Dataset: {dataset_path.name} | Run {i + 1}/{TOTAL_EVALUATION_RUNS}")

        tasks = [
                            process_single_image(
                img,
                img.with_suffix(".txt"),
                model_name,
                sem_general
            )
            for img in images_to_process
        ]

        run_results = await tqdm_asyncio.gather(*tasks, leave=False, desc=f"Processing Images (Run {i+1})")

        for res in run_results:
            if res:
                all_runs_results[res["filename"]].append(res["cer"])
                # Store detailed output only from the first run to avoid excessive data
                if i == 0:
                    detailed_outputs_for_run.append({
                        "model_name": model_name,
                        "dataset": dataset_path.name,
                        **res
                    })

    # Calculate statistics + per-run CER columns
    summary_stats = []
    for filename, cer_scores in all_runs_results.items():
        if cer_scores:
            mean_cer = np.mean(cer_scores)
            std_dev_cer = np.std(cer_scores) if len(cer_scores) > 1 else 0
            row = {
                "model_name": model_name,
                "dataset": dataset_path.name,
                "filename": filename,
                "mean_cer": mean_cer,
                "std_dev_cer": std_dev_cer
            }
            # Fill per-run CERs, padding missing runs with None
            for idx, col in enumerate(cer_run_columns):
                row[col] = cer_scores[idx] if idx < len(cer_scores) else None
            summary_stats.append(row)

    return summary_stats, detailed_outputs_for_run


async def main():
    """Main function to orchestrate the entire evaluation pipeline with checkpointing."""
    datasets = [d for d in Path(BENCHMARK_DIR).iterdir() if d.is_dir()]

    total_tasks = len(MODELS_TO_TEST) * len(datasets)
    overall_pbar = tqdm(total=total_tasks, desc="Overall Progress")

    # Dynamic per-run CER columns
    cer_run_columns = [f"cer_run_{i+1}" for i in range(TOTAL_EVALUATION_RUNS)]
    summary_fieldnames = ["model_name", "dataset", "filename", *cer_run_columns, "mean_cer", "std_dev_cer"]

    # Prepare/append CSV files with checkpoint awareness
    summary_path = Path(SUMMARY_REPORT_CSV)
    detailed_path = Path(DETAILED_OUTPUTS_CSV)

    summary_exists = summary_path.exists()
    detailed_exists = detailed_path.exists()

    existing_summary_keys = read_existing_summary_keys(summary_path)

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
                summary, details = await run_evaluation_for_model_and_dataset(model, dataset_path, overall_pbar, cer_run_columns, existing_summary_keys)

                if summary:
                    summary_writer.writerows(summary)
                    f_summary.flush()
                    # Update checkpoint set to include newly written rows
                    for row in summary:
                        existing_summary_keys.add((row["model_name"], row["dataset"], row["filename"]))
                if details:
                    detailed_writer.writerows(details)
                    f_detailed.flush()

                overall_pbar.update(1)

    overall_pbar.close()
    print(f"\nEvaluation complete. Reports saved to '{SUMMARY_REPORT_CSV}' and '{DETAILED_OUTPUTS_CSV}'.")


if __name__ == "__main__":
    asyncio.run(main()) 