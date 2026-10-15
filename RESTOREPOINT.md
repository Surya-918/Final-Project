# Storytelling Model Restore Point

**Resume coding from [`CONTINUE.md`](CONTINUE.md), not from this file.** This page is training/data background. Session handoff (what is done, what is next) lives only in `CONTINUE.md`.

## Project state

- Training entry point: `Pretrained_GPT-2.py`.
- Story API: `src/config/app.py` (legacy `src/config/test.py` remains as a compatibility wrapper).
- Persistent retrieval memory: `src/config/story_memory.py`.
- Browser conversation handling: `src/context/Context.jsx` and `src/components/Sidebar/Sidebar.jsx`.
- Existing model folders were not changed. No WritingPrompts training run has been started.
- The previous evaluation file reports perplexity `22.6228`.

## Current WritingPrompts pilot

The active Colab snapshot is `hugging face dataset/training_ready`: **12,757 train**, **677 validation**, and **683 test** rows. It filters profanity, sexual violence, selected direct insults, workshop prompts, author/editorial notes, and flagged minor-sexual content. Token length is checked with truncation disabled; over-1,024-token rows are rejected instead of clipped. Labels remain keyword candidates based on prompt and story, not verified genres; near-duplicate detection and complete semantic review are unfinished. Prior snapshots are preserved under `unused datasets`.

The stricter premise-only comparison under `unused datasets/hugging face dataset/training_ready_premise_review` has **3,401 train**, **181 validation**, and **193 test** rows. It is not the active training input because it is too small for the intended run.

## Start a fresh GPU pilot

Build `writingprompts_training_bundle.zip` from the trainer, `plot_graphs.py`, and `hugging face dataset/training_ready`, as described in [README.md](README.md). Open `train_colab.ipynb` in Colab with a GPU runtime and upload that bundle. The notebook verifies the 12,757/677/683 counts, shows three random samples per genre/split, and blocks training until manual review is confirmed. It trains on train, evaluates on validation, then calculates token loss/perplexity and next-token accuracy across all held-out test rows. It also generates from test prompts without target stories and packages the metrics as `heldout_test_metrics.json`. It sets block size 1,024, disables supplemental college data, and creates a unique empty Drive output folder.

The strict-filter set has 12,757 training rows and heuristic labels; expect reduced coverage versus the original 33k set, and note that profanity lexicons can flag contextually harmless uses. Compare generations on held-out test prompts after training. The Colab run has not been started from this local session; local PyTorch is CPU-only.

After Colab downloads the model ZIP, extract it into `trained model-college`, then point the local API at it in one PowerShell terminal:

```powershell
$env:MODEL_NAME_OR_PATH = ".\trained model-college"
npm run backend
```

Then start the web app in a second terminal with `npm run dev`.

## Retrieval memory behavior

- Each browser conversation and story has an ID, persisted in local storage and sent with every generation request.
- The API stores successful prompt/story turns in the local SQLite file `.story_memory.sqlite3`; existing databases are migrated to add story IDs.
- Retrieval only injects matching or explicitly continued story turns. An unrelated premise starts a new story ID, preventing old plots from bleeding into new prompts.
- **New Chat** creates a separate conversation. Recent restores its saved result, conversation ID, story ID, and genre without regenerating the story.
- Memory is local to this backend installation. Back up `.story_memory.sqlite3` if the story history must survive moving or reinstalling the project; it is excluded from source control.

## Validation performed

- RAG memory retrieval check passed for relevant-turn retrieval, latest-turn fallback, and conversation isolation.
- Live API regression passed: an unrelated Joker/Batman premise received a new story ID after a Raju/Shanta story, and its Gotham follow-up retrieved only the Joker-story turns.
- A live two-request API check confirmed generation and turn persistence, but the old checkpoint drifted from the story premise. The new premise/continuation training format requires the Colab GPU retrain before its generation quality can be verified.
- Python syntax checks passed for the API, memory utility, and training script.
- Colab notebook JSON and code-cell syntax checks passed; no notebook training cells were run from this local session.
- Refreshed Colab upload bundle passed ZIP integrity checks and matches the current trainer and required project files.
- Focused frontend lint passed; the existing `react/prop-types` warning was excluded because it is unrelated to the conversation-memory change.
