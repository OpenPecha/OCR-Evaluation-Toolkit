### OCR Evaluation Toolkit (Tibetan-focused)

This repository evaluates OCR quality on page images using Google Gemini models. It computes Character Error Rate (CER) against provided ground-truth transcriptions and produces summary and detailed CSV reports. Two evaluation modes are available:

- **Zero-shot evaluation**: `ocr_evaluate/evaluate_ocr.py`
- **Few-shot evaluation** (uses in-context exemplars from the same dataset): `ocr_evaluate/evaluate_ocr_fewshot.py`

Outputs include per-image CER across multiple runs, mean/std statistics, and the first-run OCR outputs for qualitative review.


### Features
- **Asynchronous, batched requests** to speed up evaluation
- **Retry with backoff** for transient API errors (e.g., HTTP 429)
- **Few-shot prompting** with K exemplars selected from each dataset
- **CSV reports**: aggregated summary and detailed first-run outputs
- **Configurable** models, concurrency, and run counts via constants in the scripts


### Requirements
- Python 3.10+
- A Google Gemini API key with access to the specified models

Install Python dependencies:

```bash
echo "Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

# Install from requirements.txt
pip install -r ocr_evaluate/requirements.txt

```

Set your Gemini API key in the environment (the code loads from `.env`):

```bash
# Create a .env in the project root or within ocr_evaluate/
cat > .env << 'EOF'
GEMINI_API_KEY=YOUR_API_KEY_HERE
EOF
```


### Data layout
Datasets live under a single benchmark directory. Each dataset is a folder containing page images (`.png` or `.jpg`) with a matching ground-truth text file of the same basename (`.txt`). Example:

```
bench mark/
  MyDataset/
    page_0001.jpg
    page_0001.txt
    page_0002.png
    page_0002.txt
```

Notes:
- The zero-shot script `evaluate_ocr.py` defaults to `benchmark/` (no space). Adjust `BENCHMARK_DIR` in that file or use the few-shot script which already points to `bench mark/`.
- Ground-truth `.txt` files must be UTF-8 and contain the target transcription for CER computation.


### How it works
- `ocr_evaluate/ocr_gemini.py`
  - Loads `GEMINI_API_KEY` from `.env`
  - Provides a synchronous `ocr_image` and an async `do_ocr` wrapper for images
- `ocr_evaluate/evaluate_ocr.py` (zero-shot)
  - Iterates datasets and images, calls Gemini
  - Repeats OCR multiple times per image to compute mean/std CER
  - Writes `evaluation_summary_report.csv` and `detailed_inference_outputs.csv`
- `ocr_evaluate/evaluate_ocr_fewshot.py`
  - Builds a few-shot prompt with K exemplars from the same dataset
  - Repeats OCR and writes `evaluation_summary_report_fewshot.csv` and `detailed_inference_outputs_fewshot.csv`


### Running
Run from the repository root so absolute imports work. Use `-m` to execute as a module so `ocr_evaluate.*` imports resolve.

Zero-shot evaluation:
```bash
python -m ocr_evaluate.evaluate_ocr
```

Few-shot evaluation:
```bash
python -m ocr_evaluate.evaluate_ocr_fewshot
```

Reports are written to the current working directory.


### Configuration
Edit the constants at the top of the scripts to customize behavior:
- **Models**: `MODELS_TO_TEST` (e.g., `gemini-2.5-flash`, `gemini-2.0-flash-lite`)
- **Concurrency**: `MAX_CONCURRENT_REQUESTS`
- **Retries/backoff**: `RETRY_ATTEMPTS`, `BACKOFF_BASE_SECONDS`, `BACKOFF_MAX_SECONDS`
- **Run count**: `TOTAL_EVALUATION_RUNS`
- Few-shot only: `FEWSHOT_K` (number of exemplars), `FEWSHOT_SEED`
- Benchmark directory: `BENCHMARK_DIR`




### Troubleshooting
- **Import error: `ocr_evaluate.ocr_gemini`**
  - Run scripts with `python -m ocr_evaluate.evaluate_ocr...` from the repo root so the package import resolves
- **HTTP 429 / rate limits**
  - The scripts backoff automatically and will retry up to the configured attempts
- **GEMINI_API_KEY missing**
  - Ensure `.env` contains `GEMINI_API_KEY` and that you run from a directory where the file can be found
- **Benchmark directory mismatch**
  - Zero-shot uses `benchmark/` by default; few-shot uses `bench mark/`. Adjust `BENCHMARK_DIR` or your directory name accordingly


### Few-shot prompt: structure and examples

The few-shot script builds a single conversation that alternates exemplar pages and their ground-truth text before asking the model to OCR the target page.

Conceptually (K=2 exemplars):
- System: "You are an absolute expert on Tibetan. Perform OCR on the provided page images. Return only clean Tibetan text, preserving natural line breaks. Do not use markdown."
- User: "Here is a similar Pecha page. Please OCR it."
- User: <Exemplar Image 1>
- Assistant: <Ground-truth text for Exemplar 1>
- User: "Here is a similar Pecha page. Please OCR it."
- User: <Exemplar Image 2>
- Assistant: <Ground-truth text for Exemplar 2>
- User: "Here is the target page. Please OCR it."
- User: <Target Image>

At runtime, K is controlled by `FEWSHOT_K` and exemplar selection is deterministic via `FEWSHOT_SEED`. Exemplars are sampled from the same dataset folder as the target image, excluding the target.

Core construction code:
```110:121:/Users/tenzingayche/Documents/synthetic-data-for-ocr/ocr_evaluate/evaluate_ocr_fewshot.py
# ... existing code ...
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
```

Sending the request with retries:
```123:133:/Users/tenzingayche/Documents/synthetic-data-for-ocr/ocr_evaluate/evaluate_ocr_fewshot.py
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
```

Quick mental model: the model first learns the expected output style from your exemplars (image -> Tibetan text), then applies that style to the final target image.

Adjust knobs in `ocr_evaluate/evaluate_ocr_fewshot.py`:
- `FEWSHOT_K`: number of exemplars to include
- `FEWSHOT_SEED`: exemplar sampling seed
- `MODELS_TO_TEST`: Gemini model(s) to call

### BDRC evaluation reference
We used the BDRC Tibetan OCR offline app during evaluation of BDRC materials: [buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app/tree/main).

### License
No license file was provided. Add one if you plan to distribute this project. 