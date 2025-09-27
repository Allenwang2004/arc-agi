from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset
import numpy as np
from copy import deepcopy
from typing import Iterable, Union


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

def prepare_evaluation_dataset(input_dataset, drop_first_train=False):
    """
    Prepares evaluation datasets from the input dataset.
    
    Args:
        input_dataset: A dataset containing 'file_name', 'train', and 'test' splits.
                       Each 'train' entry contains 'input' and 'output' examples, and
                       'test' contains 'input' and 'output' for evaluation.
        drop_first_train (bool): Whether to drop the first training example. Defaults to False. Allows the training tasks to be shortened to shorten the context length.
    
    Returns:
        - evaluation_dataset: Dataset formatted for evaluation.
    """
    
    # Evaluation dataset preparation
    evaluation_data = []
    preprompt = (
        "You are an expert in solving ARC (Abstraction and Reasoning Corpus) tasks.\n"
        "Given a series of training examples, your task is to predict the output for a\n"
        "new test example based on the same transformation rules:\n\n"
    )
    for challenge in input_dataset:
        file_name = challenge['file_name']
        train_examples = challenge['train']
        test_example = challenge['test'][0]  # Use the first test example
        
        # Use all training examples as context, optionally dropping the first one
        start_index = 1 if drop_first_train else 0
        user_message_content = (
            preprompt
        )
        for i, example in enumerate(train_examples[start_index:]):  # Include all training examples
            user_message_content += (
                f"Input:\n{np.array(example['input'])}\n"
                f"Output:\n{np.array(example['output'])}\n\n"
            )
        # Use the first test example's input as the test input
        test_input = test_example['input']
        user_message_content += f"Test Input:\n{np.array(test_input)}\nTest Output:\n"
        user_message = {"role": "user", "content": user_message_content}
        assistant_message = {
            "role": "assistant",
            "content": f"{np.array(test_example['output'])}\n\n"
        }

        evaluation_data.append({
            "file_name": file_name,
            "messages": [user_message, assistant_message]
        })
    
    # Create HuggingFace Datasets
    evaluation_dataset = Dataset.from_list(evaluation_data)
    
    return evaluation_dataset

def prepare_fine_tuning_dataset(
    input_dataset: Iterable[Dict],
    *,
    add_shuffled: Union[int, bool] = 0,   # 0/False → none, True → unlimited
    add_rotations: bool = False,
    add_mirrors: bool = False,
    omit_test: bool = True,
    apply_color_swaps: bool = False,
    num_color_swaps: int = 2,
    seed: int = 42,
) -> Dataset:
    """
    Build a HuggingFace `Dataset` for ARC fine-tuning where **every row is a
    multi-example chat prompt/response**:

        • Context  = all but one example (train examples only, or train+test
                     depending on `omit_test`)
        • Query    = that held-out example (“Test Input …”)
        • Answer   = its output grid

    Parameters
    ----------
    add_shuffled   int  – how many *extra* shuffled rows to add *per task*.
                         0/False → none.  True → as many permutations as exist.
                         For each such row we (a) choose a different held-out
                         pair, (b) randomly shuffle the context order.
    add_rotations / add_mirrors / apply_color_swaps
                   If `True`, duplicate every (possibly shuffled) row with
                   rotated / mirrored / colour-swapped variants.
    omit_test      If False ➜ held-out pair is taken from `test`;  
                   If True  ➜ held-out pair is taken from `train`.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    # ───────────────────────── helper transforms ──────────────────────────── #
    def rotate_grid(grid, angle: int):
        if angle not in (90, 180, 270):
            return grid
        return np.rot90(np.array(grid), k=angle // 90).tolist()

    def mirror_grid(grid, direction: str):
        arr = np.array(grid)
        if direction == "horizontal":
            return np.fliplr(arr).tolist()
        if direction == "vertical":
            return np.flipud(arr).tolist()
        return grid

    def apply_mapping(grid, mapping_arr: np.ndarray):
        return mapping_arr[np.array(grid)].tolist()

    # ────────────────────── build ONE prompt/response row ─────────────────── #
    def make_row(file_tag: str,
                 ctx_list: List[Dict],
                 q_pair: Dict) -> Dict:
        prompt = (
            "You are an expert in solving ARC (Abstraction and Reasoning Corpus) tasks.\n"
            "Given a series of training examples, your task is to predict the output for a\n"
            "new test example based on the same transformation rules:\n\n"
        )
        for ex in ctx_list:
            prompt += (
                f"Input:\n{np.array(ex['input'])}\n"
                f"Output:\n{np.array(ex['output'])}\n\n"
            )
        prompt += f"Test Input:\n{np.array(q_pair['input'])}\nTest Output:\n"

        return {
            "file_name": file_tag,
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": f"{np.array(q_pair['output'])}\n\n"},
            ],
        }

    # ────────────────────────── per task processing ───────────────────────── #
    def build_rows_for_task(file_name: str,
                            train_ex: List[Dict],
                            test_ex: List[Dict]) -> List[Dict]:
        """
        Build one or more **multi-example** chat rows for a single ARC task.
    
        • If `omit_test=False` the pool of possible held-out pairs is
          `test_ex + train_ex` (test examples first so the “base” row matches
          evaluation).  
        • If `omit_test=True`  the pool is `train_ex` only; every test example
          is ignored.
    
        `add_shuffled` (int or bool) controls how many *extra* rows (beyond the
        base row) are generated, each time choosing a different held-out pair
        and shuffling the order of the remaining context examples.
        """
        task_rows: List[Dict] = []
    
        # ---------- 1 – assemble pool of held-out candidates ------------------ #
        if omit_test or not test_ex:                     # ignore test examples
            if len(train_ex) < 2:                        # need ≥2 to hold one out
                return []                                # nothing we can do
            candidates = train_ex                        # held-out comes from train
        else:                                            # may hold out test *or* train
            candidates = test_ex + train_ex              # test first → base row identical
    
        # ---------- 2 – decide how many *extra* rows we want ------------------ #
        max_extra = len(candidates) - 1                  # cannot exceed pool size
        if add_shuffled is True:                         # unlimited
            want = max_extra
        else:
            want = int(add_shuffled) if add_shuffled else 0
            want = min(want, max_extra)
    
        # ---------- 3 – choose the held-out examples -------------------------- #
        chosen_candidates = [candidates[0]]              # base row
        remaining = candidates[1:]
        if want:
            rng.shuffle(remaining)
            chosen_candidates += remaining[:want]
    
        # ---------- 4 – build one row for each chosen candidate --------------- #
        for idx, held_out in enumerate(chosen_candidates):
            # context = all train examples except the held-out one (if it’s a train ex)
            #         = all train examples               (if held-out is a test ex)
            if held_out in train_ex:
                context = [ex for ex in train_ex if ex is not held_out]
            else:
                context = train_ex
    
            # shuffle context order only in the *extra* rows
            ctx_order = (rng.permutation(context).tolist()
                         if (add_shuffled and idx > 0)
                         else context)
    
            base_tag       = "" if idx == 0 else f"_shuffle{idx}"
            base_file_name = f"{file_name}{base_tag}"
    
            # -- 4a. original orientation -------------------------------------- #
            task_rows.append(make_row(base_file_name, ctx_order, held_out))
    
            # -- 4b. rotations -------------------------------------------------- #
            if add_rotations:
                for angle in (90, 180, 270):
                    ctx_rot = [{"input":  rotate_grid(ex["input"],  angle),
                                "output": rotate_grid(ex["output"], angle)}
                               for ex in ctx_order]
                    q_rot   = {"input":  rotate_grid(held_out["input"],  angle),
                               "output": rotate_grid(held_out["output"], angle)}
                    tag = f"{base_file_name}_rot{angle}"
                    task_rows.append(make_row(tag, ctx_rot, q_rot))
    
            # -- 4c. mirrors ---------------------------------------------------- #
            if add_mirrors:
                for direction in ("horizontal", "vertical"):
                    ctx_mir = [{"input":  mirror_grid(ex["input"],  direction),
                                "output": mirror_grid(ex["output"], direction)}
                               for ex in ctx_order]
                    q_mir   = {"input":  mirror_grid(held_out["input"],  direction),
                               "output": mirror_grid(held_out["output"], direction)}
                    tag = f"{base_file_name}_{direction}"
                    task_rows.append(make_row(tag, ctx_mir, q_mir))
    
        return task_rows

    # ─────────────────────────── master pipeline ──────────────────────────── #
    for challenge in input_dataset:
        file_name  = challenge["file_name"]
        base_train = challenge["train"]
        base_test  = challenge.get("test", [])

        # (1) original rows (plus rot/mirror)
        rows.extend(build_rows_for_task(file_name, base_train, base_test))

        # (2) global colour-swap augmentations
        if apply_color_swaps and num_color_swaps > 0:
            used_cols = {
                c
                for ex in base_train + base_test
                for grid in (ex["input"], ex["output"])
                for row in grid
                for c in row
            }

            swaps_done = 0
            while swaps_done < num_color_swaps:
                perm = rng.permutation(10)
                if all(perm[c] == c for c in used_cols):
                    continue                          # nothing actually swapped

                map_arr = perm
                train_swapped = [
                    {"input":  apply_mapping(ex["input"],  map_arr),
                     "output": apply_mapping(ex["output"], map_arr)}
                    for ex in deepcopy(base_train)
                ]
                test_swapped = [
                    {"input":  apply_mapping(ex["input"],  map_arr),
                     "output": apply_mapping(ex["output"], map_arr)}
                    for ex in deepcopy(base_test)
                ]

                rows.extend(
                    build_rows_for_task(
                        f"{file_name}_swap{swaps_done+1}",
                        train_swapped,
                        test_swapped,
                    )
                )
                swaps_done += 1

    return Dataset.from_list(rows)