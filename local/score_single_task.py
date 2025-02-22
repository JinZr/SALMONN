import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import tabulate
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="score parameters")
    parser.add_argument(
        "--json-in",
        type=str,
        required=True,
        help="path to the salmonn generated json file",
    )
    return parser


def cleanup_text(texts: List[str]) -> List[str]:
    return [
        text.replace("</s>", "")
        .replace("<s>", "")
        .replace("<unk>", "")
        .replace(".", "")
        .strip()
        for text in texts
    ]


def parse_text(texts: List[str]) -> Tuple[List[int], List[str]]:
    scores = []
    levels = []
    for text in texts:
        score, text = text.split(" ", maxsplit=1)
        scores.append(int(score))
        levels.append(text)
    return scores, levels


def print_task_table(title: str, score_dict: Dict[str, List]):
    print(title)
    table = []
    total_task_count = 0
    total_score_correct = 0
    total_level_correct = 0
    for task, scores in score_dict.items():
        total_task_count += scores["total"]
        total_score_correct += scores["score_correct"]
        total_level_correct += scores["level_correct"]
        table.append(
            [
                task,
                scores["total"],
                scores["score_correct"] / scores["total"],
                scores["level_correct"] / scores["total"],
            ]
        )
    table.append(
        [
            "TOTAL",
            total_task_count,
            total_score_correct / total_task_count,
            total_level_correct / total_task_count,
        ]
    )

    print(
        tabulate.tabulate(
            table, headers=["Task", "Total", "Score Accuracy", "Level Accuracy"]
        )
    )
    print()


def print_total_table(title: str, total_acc_dict: Dict[str, int]):
    print(title)
    table = []
    table.append(
        [
            total_acc_dict["total"],
            total_acc_dict["score_correct"] / total_acc_dict["total"],
            total_acc_dict["level_correct"] / total_acc_dict["total"],
        ]
    )
    print(
        tabulate.tabulate(
            table, headers=["Total", "Total Score Accuracy", "Total Level Accuracy"]
        )
    )


def main():
    args = get_parser().parse_args()
    json_in = Path(args.json_in)

    score_dict = {}
    task_acc_dict = {}
    total_acc_dict = {"level_correct": 0, "score_correct": 0, "total": 0}
    with open(json_in) as f:
        data = json.load(f)

    for batch in tqdm(data):
        tasks = batch["task"]
        texts = batch["text"]
        ground_truths = batch["ground_truth"]

        texts = cleanup_text(texts)
        scores, texts = parse_text(texts)

        ground_truths = cleanup_text(ground_truths)
        ground_truth_scores, ground_truth_texts = parse_text(ground_truths)

        assert len(scores) == len(ground_truth_scores)
        assert len(texts) == len(ground_truth_texts)

        for i, (task, score, text, ground_truth_score, ground_truth_text) in enumerate(
            zip(tasks, scores, texts, ground_truth_scores, ground_truth_texts)
        ):
            if task not in score_dict:
                score_dict[task] = []
            if task not in task_acc_dict:
                task_acc_dict[task] = {
                    "level_correct": 0,
                    "score_correct": 0,
                    "total": 0,
                }
            score_dict[task].append(
                {
                    "score": score,
                    "text": text,
                    "ground_truth_score": ground_truth_score,
                    "ground_truth_text": ground_truth_text,
                }
            )
            task_acc_dict[task]["total"] += 1
            if score == ground_truth_score:
                task_acc_dict[task]["score_correct"] += 1
            if text == ground_truth_text:
                task_acc_dict[task]["level_correct"] += 1

            total_acc_dict["total"] += 1
            if score == ground_truth_score:
                total_acc_dict["score_correct"] += 1
            if text == ground_truth_text:
                total_acc_dict["level_correct"] += 1
    print_task_table("Task Accuract Table", task_acc_dict)
    # print_total_table("Total Accuracy Table", total_acc_dict)


if __name__ == "__main__":
    main()
