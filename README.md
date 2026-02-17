# Kaggle/GitHub/Hugging Face Notebook Analyzer

Analyze public notebooks and rank the most commonly used Python libraries (unique per notebook). Supports Kaggle, GitHub, and Hugging Face as separate data sources.

## Requirements

- Python 3.12+
- Kaggle API (for Kaggle source): `pip install kaggle`
- Plotly (optional, for charts): `pip install plotly`

## Setup

### Kaggle credentials (required for Kaggle source)

1. Create a Kaggle API token from your account settings.
2. Save it as `~/.kaggle/kaggle.json`.

### Optional tokens (recommended for higher limits)

- GitHub: `export GITHUB_TOKEN="..."`
- Hugging Face: `export HF_TOKEN="..."`

## Usage

Run the tool from the project root:

```bash
python3 main.py [options]
```

### Common options

- `--source`: `kaggle` | `github` | `huggingface` (default: `kaggle`)
- `--max-notebooks`: number of notebooks to analyze (default: `50`)
- `--output-dir`: base output directory (default: `data`)

### Kaggle source

```bash
python3 main.py --source kaggle --selection top --max-notebooks 50
```

Options:

- `--selection`: `top` | `latest` | `hotness`
- `--sort-by`: overrides selection sort (e.g., `voteCount`, `dateCreated`, `viewCount`)
- `--competition`: filter by competition slug

Downloads are stored under:

```
data/kaggle_notebooks/<selection>/
```

### GitHub source (most-starred notebooks)

```bash
python3 main.py --source github --max-notebooks 50
```

Options:

- `--github-query`: GitHub code search query (default: `extension:ipynb`)

Downloads are stored under:

```
data/github_notebooks/most-starred/
```

### Hugging Face source (most-liked by default)

```bash
python3 main.py --source huggingface --max-notebooks 50
```

Options:

- `--hf-repo-types`: comma-separated list of repo types (default: `model,dataset,space`)
- `--hf-sort`: `likes`, `downloads`, `trending_score`, `created_at`, `last_modified` (default: `likes`)

Downloads are stored under:

```
data/huggingface_notebooks/<selection>/<repo_type>/
```

## Output

The tool prints a ranked list of libraries and, when detectable, Jupyter extensions. Library charts are displayed with Plotly if installed. Hugging Face runs also render per-repo-type charts (model/dataset/space) when available.

Example plot title:

```
Most used python libraries (unique per notebook) Analyzed 50 most-liked huggingface notebooks
```

## Dash dashboard

Run the dashboard after you have downloaded notebooks with `main.py`.

```bash
pip install dash "plotly[express]"
python3 dashboard.py --data-dir data
```

Optional flags:

- `--max-items`: max bars per chart (default: 20)
- `--max-notebooks`: limit notebooks analyzed per selection (default: 0 = no limit)

The dashboard reads from:

```
data/kaggle_notebooks/
data/github_notebooks/
data/huggingface_notebooks/
```

Each source has its own tab with library usage charts plus full tables for libraries and extensions. The Hugging Face tab lets you switch between repo types (model/dataset/space), and Kaggle/Hugging Face include an "All" option to aggregate their subcategories. A fourth tab aggregates all downloaded notebooks across sources.

### Run dashboard in Docker (data bundled in image)

```bash
podman run --rm -p 8051:8051 quay.io/rh_ee_atheodor/lib-analysis-dashboard:v1
```

Open: `http://localhost:8051`


### Build 
```bash
podman build -t quay.io/rh_ee_atheodor/lib-analysis-dashboard:v1 .
podman push quay.io/rh_ee_atheodor/lib-analysis-dashboard:v1
```

## Notes

- Each library is counted once per notebook.
- Submodules and aliases are normalized (e.g., `sklearn.model_selection` -> `scikit-learn`).
- Invalid or non-notebook files are skipped with a warning.
- If Plotly is missing, the tool still prints results.

## Troubleshooting

- Kaggle 429 errors: reduce `--max-notebooks` or retry later.
- GitHub/HF rate limits: set `GITHUB_TOKEN` or `HF_TOKEN`.
- Stuck run: the fetch can take time. Progress logs are printed for Hugging Face; allow a few minutes for large limits.
