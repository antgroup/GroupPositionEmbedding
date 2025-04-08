import os
import json
from typing import Sequence
import textdistance
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics import precision_recall_fscore_support
import argparse


def calculate_f1(predictions, ground_truth):
    # Convert ground truth format to match predictions
    ground_truth_flat = [gt[0].lower() for gt in ground_truth]
    predictions = [pred.lower() for pred in predictions]

    # Compute matches between predictions and ground truths
    matched = [pred == gt for pred, gt in zip(predictions, ground_truth_flat)]

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        matched, [True] * len(matched), average='binary'
    )

    # Identify incorrect predictions
    incorrect_predictions = [
        (pred, gt) for pred, gt, match in zip(predictions, ground_truth_flat, matched) if not match
    ]
    return f1


def read_jsonl(file_path):
    """
    Reads a JSONL file from the given path and returns a list of objects.

    Parameters:
    file_path (str): The path to the JSONL file.

    Returns:
    list: A list of objects, where each object is a JSON object from a line in the JSONL file.
    """
    objects = []
    with open(file_path, 'r', encoding='utf-8') as f_:
        for line in f_:
            try:
                obj = json.loads(line.strip())
                objects.append(obj)
            except json.JSONDecodeError:
                pass
    return objects


def eval_anls(out_items, ref_items, threshold=0.5):
    scores = []
    for out, ref in zip(out_items, ref_items):
        val = out.lower()
        possible_vals = [item.lower() for item in ref]
        best_score = max([textdistance.levenshtein.normalized_similarity(val, pos) for pos in possible_vals])
        if 1 - threshold >= best_score:
            best_score = 0.0
        scores.append(best_score)
    return sum(scores) / (len(scores) + 1e-20)


def eval_in_accuracy(out_items, ref_items):
    scores = []
    for out, ref in zip(out_items[::-1], ref_items[::-1]):
        if ref.lower() in out.lower():
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores) / len(scores)


def eval_equal_accuracy(out_items, ref_items):
    scores = []
    for out, ref in zip(out_items[::-1], ref_items[::-1]):
        if ref.lower() == out.lower():
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores) / len(scores)


def eval_cider(predictions: Sequence[str], targets: Sequence[Sequence[str]]) -> float:
    """Compute CIDEr score."""
    coco_tokenizer = PTBTokenizer()
    scorer = Cider()
    gts = {str(i): target for i, target in enumerate(targets)}
    res = {str(i): [prediction] for i, prediction in enumerate(predictions)}

    score, scores = scorer.compute_score(gts=gts, res=res)
    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores.tolist()]
    return score, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--input_dir', type=str, help="Directory containing prediction files")
    args = parser.parse_args()

    # Handle input directory
    if not args.input_dir:
        args.input_dir = input("Please enter the directory containing prediction files: ").strip()
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Provided directory '{args.input_dir}' does not exist.")

    dataset_names = ['forms', 'newspapers', 'slides', 'synthtables', 'synthdocs', 'websites']

    print_list = ['|dataset|forms|newspapers|slides|synthtables|synthdocs|websites|', '|---|---|---|---|---|---|---|']
    score_string = "|model|"

    for dataset_name in dataset_names:
        input_path = os.path.join(args.input_dir, dataset_name + '.jsonl')
        if not os.path.exists(input_path):
            score_string += "|"
            continue

        input_data = read_jsonl(input_path)
        if not input_data:
            score_string += "|"
            continue

        refs = []
        all_refs = []
        outs = []

        for sample in input_data:
            if 'outputs' not in sample:
                continue

            out_dict = sample['model_output'] if 'output' not in sample['model_output'] else sample['model_output']['output']
            questions, answers = sample['inputs'], sample['outputs']

            if 'all_outputs' in sample:
                all_answers = sample['all_outputs']
            else:
                all_answers = [[item] for item in sample['outputs']]

            for qidx, question in enumerate(questions):
                if question in out_dict:
                    outs.append(out_dict[question])
                    refs.append(answers[qidx])
                    all_refs.append(all_answers[qidx])

        if not outs or not refs or not all_refs:
            score_string += "|"
            continue

        anls_score = eval_anls(outs, all_refs)
        cider_score, _ = eval_cider(outs, all_refs)
        anls_score, cider_score = round(anls_score * 100, 2), round(cider_score, 2)

        if dataset_name == 'visualMRC':
            score_string += f"{anls_score}/{cider_score}|"
        else:
            score_string += f"{anls_score}|"

    print_list.append(score_string)

    for line in print_list:
        print(line)