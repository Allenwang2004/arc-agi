from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset

__all__ = [
    "get_arc_datasets",
]

###############################################################################
# Internal helpers
###############################################################################

def _split_dictionary(data: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Split tasks that have *multiple* test IO pairs into separate entries.

    Each new key is suffixed with an integer index (e.g. `"abcd1234_1"`).

    Returns the revamped dictionary **and** a list with the names of the
    expanded tasks (useful for debugging).
    """
    result: Dict[str, Dict] = {}
    split_files: List[str] = []

    for key, value in data.items():
        test_list = value.get("test", [])
        train_list = value.get("train", [])

        if len(test_list) > 1:
            for idx, test_item in enumerate(test_list):
                new_key = f"{key}_{idx}"
                result[new_key] = {"test": [test_item], "train": train_list}
                split_files.append(new_key)
        else:
            # untouched – already a single‑case task
            result[key] = value

    return result, split_files


def _build_dataframe(
    *,
    challenges: Dict[str, Dict],
    solutions: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    """Construct the canonical DataFrame used for both splits."""
    data_rows: List[Dict] = []

    for file_name, grids in challenges.items():
        train_grids = grids.get("train", [])
        test_inputs = grids.get("test", [])

        # If solutions are provided (train and evaluation splits) we harvest
        # the correct test outputs. When we created `challenges` we already
        # fan‑out multi‑case tasks, so every entry has exactly one test item.
        if solutions is not None:
            parts = file_name.split("_")
            base_key, test_idx = parts[0], int(parts[1] if len(parts) > 1 else 0)
            correct_outputs = solutions.get(base_key, [])
            # Guard: some evaluation sets may intentionally omit solutions
            if test_idx >= len(correct_outputs):
                raise ValueError(
                    f"No solution available for {file_name} (idx {test_idx})."
                )
            test_output = [{"output": correct_outputs[test_idx]}]
        else:
            test_output = []  # unknown at inference time

        combined_test = (
            [
                {
                    "input": test_inputs[0]["input"],
                    "output": test_output[0]["output"],
                }
            ]
            if test_output
            else test_inputs
        )

        data_rows.append(
            {
                "file_name": file_name,
                "train": train_grids,
                "test_input": test_inputs,
                "test_output": test_output,
                "test": combined_test,
            }
        )

    return pd.DataFrame(data_rows)


def _apply_subsample(df: pd.DataFrame, subsample_file: Path | None) -> pd.DataFrame:
    """Optionally filter rows by problem IDs listed in *subsample_file*."""
    if subsample_file is None:
        return df

    with subsample_file.open() as fp:
        ids = [line.strip() for line in json.load(fp)]  # expects a JSON list

    # The DataFrame rows may be *split* versions – we harvest the *root* ID
    return df[df["file_name"].str.extract(r"^([a-f0-9]+)")[0].isin(ids)]


###############################################################################
# Public API
###############################################################################

def get_arc_datasets(
    *,
    data_dir: str | Path = "/Users/coconut/arc-agi/arc-prize-2025",  # directory holding the arc‑agi_*.json files
    eval_subsample_json: str | Path | None = None,
    train_subsample_json: str | Path | None = None,
) -> Tuple[Dataset, Dataset]:
    """Return `(arc_train, arc_eval)` as `datasets.Dataset` objects.

    Parameters
    ----------
    data_dir:
        Base directory containing the official ARC‑AGI JSON files (training,
        evaluation). Defaults to the current working directory.

    eval_subsample_json / train_subsample_json:
        Optional path(s) to a JSON file with a list of problem IDs that
        should be kept. When *None* the full split is used.

    Notes
    -----
    *We purposely do not expose the private‑test split here – that file set
    lacks ground‑truth solutions and must be handled separately.*
    """
    data_dir = Path(data_dir)

    ###############  TRAIN SPLIT  ###########################################
    with (data_dir / "arc-agi_training_challenges.json").open() as fp:
        train_challenges = json.load(fp)
    train_challenges, _ = _split_dictionary(train_challenges)

    with (data_dir / "arc-agi_training_solutions.json").open() as fp:
        train_solutions = json.load(fp)

    train_df = _build_dataframe(
        challenges=train_challenges,
        solutions=train_solutions,
    )
    train_df = _apply_subsample(train_df, Path(train_subsample_json) if train_subsample_json else None)

    ###############  EVAL SPLIT  ############################################
    with (data_dir / "arc-agi_evaluation_challenges.json").open() as fp:
        eval_challenges = json.load(fp)
    eval_challenges, _ = _split_dictionary(eval_challenges)

    # Evaluation split *does* ship with solutions – they just aren’t public on
    # Kaggle. Adjust the path below if you keep them elsewhere.
    eval_sol_path = data_dir / "arc-agi_evaluation_solutions.json"
    if eval_sol_path.exists():
        with eval_sol_path.open() as fp:
            eval_solutions = json.load(fp)
    else:
        eval_solutions = None  # e.g. you’re working on the competition

    eval_df = _build_dataframe(
        challenges=eval_challenges,
        solutions=eval_solutions,
    )
    eval_df = _apply_subsample(eval_df, Path(eval_subsample_json) if eval_subsample_json else None)

    ###############  CONVERT → HF DATASET  ###################################
    arc_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    arc_eval = Dataset.from_pandas(eval_df.reset_index(drop=True))

    return arc_train, arc_eval