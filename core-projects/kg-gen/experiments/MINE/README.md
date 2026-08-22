# Running MINE

MINE now automatically loads evaluation data from Hugging Face, making it easier to run evaluations without managing local files.

## Quick Start

### 1. Run Evaluation

1. Set your OpenAI API key as an environment variable:
   - **Windows PowerShell:** `$env:OPENAI_API_KEY="your_actual_key_here"`
   - **Linux/Mac:** `export OPENAI_API_KEY="your_actual_key_here"`

2. Run the evaluation script:
   ```bash
   python _1_evaluation.py --model openai/gpt-5-nano --evaluation-model local
   ```

3. Results will be saved in `results/{model-config}/`:
   - `results_{i}.json` - Evaluation results for each essay
   - `kg_{i}.json` - Generated knowledge graph for each essay

### 2. Compare Results

Generate comprehensive comparison charts and statistics:
```bash
python _2_compare_results.py
```

This creates:
- `results/results.png` - Comprehensive comparison plot
- `results/summary.txt` - Detailed statistics and rankings
- `results/comparisons/` - Pairwise comparison plots

### 3. Result Reports

`_3_visualize.py` is a headless CLI report generator (no web server):
```bash
python _3_visualize.py                      # all models, all essays
python _3_visualize.py --essays 0 1 --export-kg
```

It writes to `results/reports/`:
- `essay_{i}_results.csv` - Per-query retrieved context and evaluation for each model
- `accuracy_by_essay.png` - Grouped bar chart of accuracy per model per essay

With `--export-kg` it also writes `results/{model}/kg_{i}_visualization.html` for each
knowledge graph. Use `--list-models` to see available result directories and `--show`
to display the chart interactively instead of only saving it.

> Interactive UI: the product interface is **BubbleLab (TypeScript)**, located at
> `core-projects/BubbleLab`. There is no Python web dashboard in this repo.

## Data Loading

The evaluation script automatically:
- ✅ **Downloads evaluation data from Hugging Face** ([kg-gen-evaluation-answers](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-answers))
- ✅ **Falls back to local files** if Hugging Face is unavailable
- ✅ **Shows clear status messages** about data source

**Source Essays:** Available at [kg-gen-evaluation-essays](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays) - use these to generate your knowledge graphs.

## Local Development

If you prefer to use local files or Hugging Face is unavailable:
- Ensure [`answers.json`](answers.json) exists with the evaluation questions and answers
- The script will automatically detect and use local files as fallback