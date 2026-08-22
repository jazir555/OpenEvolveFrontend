#!/usr/bin/env python3
"""
Offline reporting for knowledge graph evaluation results.

Loads MINE evaluation results from disk (downloading them once if needed),
writes per-essay result tables to CSV, saves accuracy charts with matplotlib,
and can export interactive knowledge graph HTML files via KGGen.visualize.

The interactive product UI lives in BubbleLab (TypeScript); this script is a
headless CLI for local analysis only.
"""

import argparse
import json
from pathlib import Path

import matplotlib

import pandas as pd
from datasets import load_dataset
from kg_gen import KGGen
from kg_gen.models import Graph
import urllib.request
import zipfile

RESULTS_URL = "https://github.com/stair-lab/kg-gen/releases/download/MINE-evaluations-expanded/results.zip"
RESULTS_DIR = Path("experiments/MINE/results")
REPORTS_DIR = RESULTS_DIR / "reports"


def ensure_results_exist():
    """Download and extract results if the directory doesn't exist or is empty."""
    # Check if results directory exists and has content
    if RESULTS_DIR.exists():
        # Check if directory has any subdirectories (model results)
        subdirs = [
            d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != "comparisons"
        ]
        if subdirs:
            return  # Results already exist

    # Results don't exist or directory is empty - download them
    print("Downloading evaluation results (first time setup)...")

    try:
        # Create MINE directory if it doesn't exist
        mine_dir = Path("experiments/MINE")
        mine_dir.mkdir(parents=True, exist_ok=True)

        # Download the zip file
        zip_path = mine_dir / "results.zip"

        print("Downloading results.zip...")
        urllib.request.urlretrieve(RESULTS_URL, zip_path)

        # Extract the zip file
        print("Extracting results...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(mine_dir)

        # Clean up the zip file
        zip_path.unlink()

        print("[OK] Results downloaded and extracted successfully!")

    except Exception as e:
        print(f"[FAIL] Failed to download results: {e}")
        print("Please manually download from: " + RESULTS_URL)
        raise


def load_hf_dataset():
    """Load the HuggingFace dataset."""
    dataset = load_dataset("josancamon/kg-gen-MINE-evaluation-dataset")["train"]
    return dataset.to_list()


def discover_result_directories(results_folder=None):
    """Discover all directories in the results folder."""
    if results_folder is None:
        results_folder = RESULTS_DIR
    results_path = Path(results_folder)
    if not results_path.exists():
        return []

    directories = [d for d in results_path.iterdir() if d.is_dir()]
    directories.sort()
    return [d.name for d in directories]


def load_all_results(results_folder=None):
    """Load all results from all directories."""
    if results_folder is None:
        results_folder = RESULTS_DIR
    results_path = Path(results_folder)
    all_results = {}

    if not results_path.exists():
        return all_results

    directories = [d for d in results_path.iterdir() if d.is_dir()]

    for directory in directories:
        model_name = directory.name
        model_results = {}

        json_files = sorted(directory.glob("results_*.json"))
        for json_file in json_files:
            try:
                idx = int(json_file.stem.split("_")[1])
                with open(json_file, "r") as f:
                    data = json.load(f)
                    # Remove accuracy summary if present
                    if isinstance(data, list) and len(data) > 0:
                        if (
                            isinstance(data[-1], dict)
                            and "accuracy" in data[-1]
                            and len(data[-1]) == 1
                        ):
                            data = data[:-1]
                    model_results[idx] = data
            except Exception as e:
                print(f"Warning: could not read {json_file}: {e}")

        if model_results:
            all_results[model_name] = model_results

    return all_results


def load_kg_file(model_name: str, essay_idx: int, results_folder=None):
    """Load knowledge graph file for a specific model and essay."""
    if results_folder is None:
        results_folder = RESULTS_DIR
    kg_path = Path(results_folder) / model_name / f"kg_{essay_idx}.json"
    if kg_path.exists():
        with open(kg_path, "r") as f:
            kg_data = json.load(f)
        return Graph(**kg_data)
    return None


def export_kg_visualization(
    graph: Graph, model_name: str, essay_idx: int, open_in_browser: bool = False
):
    """Write a standalone HTML visualization of a knowledge graph to disk."""
    if graph is None or not graph.entities:
        print(f"[WARN] No knowledge graph available for {model_name}")
        return None

    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"kg_{essay_idx}_visualization.html"

    KGGen.visualize(graph, str(output_path), open_in_browser=open_in_browser)
    print(
        f"[OK] {model_name} essay {essay_idx}: "
        f"{len(graph.entities)} entities, {len(graph.relations)} relations "
        f"-> {output_path}"
    )
    return output_path


def build_essay_table(dataset, all_results, essay_idx: int, model_names):
    """Build a query-by-model table of retrieved contexts and evaluations."""
    essay_data = dataset[essay_idx]
    queries = essay_data.get("generated_queries", [])

    table_data = []
    for query_idx, query in enumerate(queries):
        row = {
            "Query #": query_idx + 1,
            "Query": query,
        }

        for model_name in model_names:
            model_results = all_results.get(model_name, {}).get(essay_idx, [])

            if query_idx < len(model_results):
                result = model_results[query_idx]
                row[f"{model_name} - Context"] = result.get("retrieved_context", "N/A")
                row[f"{model_name} - Evaluation"] = (
                    "correct" if result.get("evaluation", 0) == 1 else "incorrect"
                )
            else:
                row[f"{model_name} - Context"] = "N/A"
                row[f"{model_name} - Evaluation"] = "N/A"

        table_data.append(row)

    return pd.DataFrame(table_data)


def essay_accuracy(all_results, essay_idx: int, model_name: str):
    """Percentage of correctly answered queries for one model/essay pair."""
    results = all_results.get(model_name, {}).get(essay_idx, [])
    if not results:
        return None
    correct = sum(1 for r in results if r.get("evaluation", 0) == 1)
    return 100.0 * correct / len(results)


def plot_accuracies(all_results, essay_indices, model_names, output_path, show=False):
    """Save a grouped bar chart of per-essay accuracy for each model."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(essay_indices)), 6))
    x = np.arange(len(essay_indices))
    width = min(0.8 / max(len(model_names), 1), 0.35)

    for i, model_name in enumerate(model_names):
        values = [
            essay_accuracy(all_results, idx, model_name) or 0.0 for idx in essay_indices
        ]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xlabel("Essay index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MINE evaluation accuracy by essay")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([str(idx) for idx in essay_indices])
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[OK] Accuracy chart saved to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model result directories to include (default: all discovered)",
    )
    parser.add_argument(
        "--essays",
        nargs="*",
        type=int,
        default=None,
        help="Essay indices to report on (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help=f"Directory for CSV/PNG output (default: {REPORTS_DIR})",
    )
    parser.add_argument(
        "--export-kg",
        action="store_true",
        help="Also write kg_{essay}_visualization.html for each model/essay",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open exported knowledge graph HTML files in a browser",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the accuracy chart in a window instead of only saving it",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered model result directories and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    ensure_results_exist()

    if args.list_models:
        for name in discover_result_directories():
            print(name)
        return

    all_results = load_all_results()
    if not all_results:
        print(f"[FAIL] No results found in {RESULTS_DIR}")
        return

    model_names = sorted(all_results.keys())
    if args.models:
        unknown = [m for m in args.models if m not in all_results]
        if unknown:
            print(f"[WARN] Unknown models ignored: {', '.join(unknown)}")
        model_names = [m for m in args.models if m in all_results]

    if not model_names:
        print("[FAIL] No matching models to report on.")
        return

    available_essays = sorted(
        {idx for name in model_names for idx in all_results[name]}
    )
    essay_indices = args.essays or available_essays
    missing = [idx for idx in essay_indices if idx not in available_essays]
    if missing:
        print(f"[WARN] No results for essays: {missing}")
    essay_indices = [idx for idx in essay_indices if idx in available_essays]

    if not essay_indices:
        print("[FAIL] No essay results found for selected models.")
        return

    dataset = load_hf_dataset()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for essay_idx in essay_indices:
        essay_data = dataset[essay_idx]
        topic = essay_data.get("essay_topic", "Unknown")
        print(f"\nEssay {essay_idx}: {topic}")

        for model_name in model_names:
            accuracy = essay_accuracy(all_results, essay_idx, model_name)
            if accuracy is None:
                print(f"  {model_name}: no results")
            else:
                print(f"  {model_name}: {accuracy:.2f}%")

        df = build_essay_table(dataset, all_results, essay_idx, model_names)
        if df.empty:
            print("  [WARN] No queries found for this essay.")
        else:
            csv_path = output_dir / f"essay_{essay_idx}_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  [OK] Table saved to {csv_path}")

        if args.export_kg:
            for model_name in model_names:
                graph = load_kg_file(model_name, essay_idx)
                if graph is None:
                    continue
                export_kg_visualization(
                    graph, model_name, essay_idx, open_in_browser=args.open_browser
                )

    plot_accuracies(
        all_results,
        essay_indices,
        model_names,
        output_dir / "accuracy_by_essay.png",
        show=args.show,
    )


if __name__ == "__main__":
    main()
