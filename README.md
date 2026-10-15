# Narrative-Hub
Narrative Hub is an AI-powered tool help people to generate creative and sharing stories from small prompt.

developed a Storytelling AI, which leverages a pretrained GPT-2 model to generate engaging and creative stories.
The main goal of the project was to explore the capabilities of AI in creative writing and provide users with a platform to generate unique narratives.

## Run the app locally

To use the college-trained model, extract `storytelling-college-model.zip` from Colab into a project-root folder named `trained model-college`. In the backend PowerShell terminal, set the model path before starting the API:

```powershell
$env:MODEL_NAME_OR_PATH = ".\trained model-college"
npm run backend
```

Keep that terminal open, then run `npm run dev` in a second terminal. The API loads the selected model folder.

The React app and story-generation API run separately. In the project directory, create and activate a Python environment, install the API requirements, and start the backend:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r src/config/requirements.txt
npm run backend
```

Keep that terminal open. In a second terminal, run `npm run dev` and open the URL Vite prints. The API prefers `trained model-college`, then other available local model folders; it falls back to base GPT-2 only when no local model folder is available.

## Deploy the app (Vercel + Railway)

Vercel hosts the React/Vite frontend; Railway runs the FastAPI service and the
GPT-2 model. The trained weights are about 498 MB, so keep them out of the
frontend bundle and store them on a Railway persistent volume. The API also
stores story memory on that volume.

### 1. Deploy the API and model on Railway

1. Push this project directory to a GitHub repository and create a Railway
   project from that repository. Railway will build the included `Dockerfile`.
2. Attach a persistent volume to the Railway service and set its mount path to
   `/data`.
3. In Railway's service variables, set:

   ```text
   MODEL_NAME_OR_PATH=/data
   STORY_MEMORY_DB=/data/story_memory.sqlite3
   CORS_ORIGINS=*
   ```

4. Upload the **contents** of the local `trained model-college` folder directly
   into the volume root (`/data`). At minimum, include `config.json`,
   `model.safetensors`, and the tokenizer files (`tokenizer.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`,
   `vocab.json`, and `merges.txt`, when present). Do not put them in an extra
   nested folder.
5. Generate a public Railway domain. Check `https://<railway-domain>/health`;
   the response should show the selected model path as `/data`.

### 2. Deploy the frontend on Vercel

1. Import the same GitHub repository in Vercel. Select Vite if asked; use
   `npm run build` as the build command and `dist` as the output directory.
2. Add the build-time environment variable
   `VITE_API_URL=https://<railway-domain>` and deploy.
3. Copy the deployed Vercel origin (for example, `https://your-app.vercel.app`)
   into Railway's `CORS_ORIGINS` variable, replacing `*`, then redeploy the
   Railway service.

`/data` must remain mounted for the model files and story-memory database to
survive Railway restarts. Allocate at least 2 GB of service memory (4 GB is
safer); the API uses CPU inference, so the first generation may take longer
while the model loads. Do not commit `.env.local` or model weights to the
frontend repository.

Key Features and Achievements:

Advanced AI Storytelling: Utilized the GPT-2 model to generate coherent and captivating stories based on user input.
Interactive Web Interface: Developed a user-friendly website using React.js, allowing users to interact with the AI seamlessly.
Robust Backend Integration: Employed Flask and FastAPI to create a scalable and efficient backend that handles AI requests and responses smoothly.
Customizability: Enabled users to customize story parameters, such as genre and character names, to tailor the generated stories to their preferences.
Real-time Story Generation: Achieved near real-time story generation, providing an interactive and engaging user experience.

here's you can watch our Demo:
https://youtu.be/QhK4bdBxJug

if you need any help feel free to contact us on: NarrativeHub@outlook.com

## Recommended training path: WritingPrompts on Colab

The active Colab snapshot is `hugging face dataset/training_ready`: **12,757 train**, **677 validation**, and **683 test** rows. It filters profanity from the `better-profanity` lexicon, sexual-violence terms, selected direct insults, author/editorial notes, and flagged sexual content near minors, plus existing short/noisy pairs. The actual GPT-2 tokenizer is used with truncation disabled during length checks; rows over 1,024 tokens are rejected, never silently clipped. Labels remain prompt-plus-story keyword candidates, not verified genres. Near-duplicate detection and full semantic review are not complete. Previous snapshots are preserved under `unused datasets`.

From the project root, build the dedicated bundle:

```powershell
$bundleItems = @(
   ".\Pretrained_GPT-2.py",
   ".\plot_graphs.py",
   ".\hugging face dataset\training_ready"
)
Compress-Archive -Path $bundleItems -DestinationPath ".\writingprompts_training_bundle.zip" -Force
```

Open [train_colab.ipynb](train_colab.ipynb) in Google Colab, select a GPU runtime, upload `writingprompts_training_bundle.zip`, and run the cells in order. The notebook checks the 12,757/677/683 split counts, displays three random examples per genre and split, and blocks training until manual review is confirmed. It trains only on `train.txt`, evaluates on `validation.txt`, then computes token loss/perplexity and next-token accuracy over all held-out test rows. Separate test generations receive prompts only, not target stories. Test metrics are saved as `heldout_test_metrics.json`. Training uses a 1,024-token block with a low-memory microbatch and writes to a unique empty Drive folder; it does not reuse an old checkpoint. The trainer masks the `Idea:`/`Story:` prefix from training and evaluation loss so metrics focus on story continuation tokens.

Strict language filtering leaves a much smaller **12.8k** set, so expect lower coverage. The profanity lexicon may also flag contextually harmless matches; review samples before training. Afterward, review Action and Drama completions on held-out test prompts for premise alignment, coherence, and endings; token metrics alone do not prove story quality. Check source terms and rights before public release.

The older college/Gutenberg training material below is historical or optional. It is not the recommended WritingPrompts training path.

### Optional: prepare downloaded Gutenberg books for action/drama training

The 81 selected Project Gutenberg books in `unused datasets/newly dataset/books` can be
converted to the trainer's one-example-per-line format without modifying the
original downloads. If any configured source file is missing, download the
configured books first:

```powershell
.\.venv\Scripts\python.exe tools\download_gutenberg_books.py
.\.venv\Scripts\python.exe tools\prepare_gutenberg_books.py
```

This writes `gutenberg_train.txt`, `gutenberg_validation.txt`, and
`manifest.json` under `unused datasets/newly dataset/books/prepared`. The manifest records each
book's source URL, provisional genre assignment, split, and chunk counts.
Validation is split by whole book to avoid putting excerpts from one book in
both splits. The adventure books are used as an **action proxy**, and the
literary novels as a **drama proxy**; neither set has verified action/drama
labels. The 81 works provide broader source coverage, but the older prose
style and proxy labels do not guarantee improvement in a modern story generator.

### Generate the stricter premise-only comparison subset

Keep the source Hugging Face `.parquet` files under
`unused datasets/hugging face dataset/source parquet`.
From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m pip install duckdb tokenizers better-profanity
.\.venv\Scripts\python.exe tools\segregate_writingprompts.py --input-dir "unused datasets\hugging face dataset\source parquet" --output-dir "unused datasets\hugging face dataset\segregated_premise_review"
.\.venv\Scripts\python.exe tools\prepare_writingprompts_training.py --input-dir "unused datasets\hugging face dataset\segregated_premise_review" --output-dir "unused datasets\hugging face dataset\training_ready_premise_review" --tokenizer "trained model-college"
```

This optional stricter experiment requires at least two distinct keyword cues in the premise; story-body keywords cannot assign its label. It also uses the cleaner's meta-noise filters, untruncated GPT-2 token check, and exact prompt/story deduplication. It does not detect near-duplicate stories or semantically verify premise/story alignment. Current output is **3,401 train**, **181 validation**, and **193 test** rows. Counts and filter reasons are in `preparation_report.json`.

This comparison subset is too small for the active run. Keep validation/test held out, review samples before training, and check source terms and rights before sharing the dataset or model.

## Story memory and restore point

Story turns are kept locally in `.story_memory.sqlite3` and retrieved within their story. An unrelated new premise automatically starts an isolated story memory; **New Chat** starts a new conversation, and Recent restores a saved story. Active conversation, story ID, and genre survive browser reloads. The database is local and excluded from source control. See [RESTOREPOINT.md](RESTOREPOINT.md) for training data notes and recovery steps.
