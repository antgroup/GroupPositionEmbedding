import os
import json
from typing import Sequence

import textdistance
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

from sklearn.metrics import precision_recall_fscore_support

def calculate_f1(predictions, ground_truth):
    # 将标准答案的格式转换为与预估结果的格式一致
    ground_truth_flat = [gt[0].lower() for gt in ground_truth]
    predictions = [pred.lower() for pred in predictions]

    # 计算预测值与标准答案之间的匹配情况
    matched = [pred == gt for pred, gt in zip(predictions, ground_truth_flat)]

    # 计算精度、召回率和 F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        matched, [True]*len(matched), average='binary'
    )
    
    # 找出错误的预测
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
            except json.JSONDecodeError as e:
                pass
                # print(f"Error decoding JSON on line: {line.strip()}")
                # print(f"Error: {e}")
    return objects


def eval_anls(out_items, ref_items, threshold=0.5):
    _scores = []

    for out, ref in zip(out_items, ref_items):
        val = out.lower()
        possible_vals = [item.lower() for item in ref]
        best_score = max([textdistance.levenshtein.normalized_similarity(val, pos)
            for pos in possible_vals])
        if 1 - threshold >= best_score:
            best_score = 0.0
        _scores.append(best_score)    

    return sum(_scores) / (len(_scores) + 1e-20)


def eval_in_accuracy(out_items, ref_items):
    _scores = []
    for out, ref in zip(out_items[::-1], ref_items[::-1]):
        if ref.lower() in out.lower():
            ans = 1
        else:
            ans = 0
        _scores.append(ans)
    return sum(_scores) / len(_scores)


def eval_equal_accuracy(out_items, ref_items):
    _scores = []
    for out, ref in zip(out_items[::-1], ref_items[::-1]):
        if ref.lower() == out.lower():
            ans = 1
        else:
            ans = 0
        _scores.append(ans)
    return sum(_scores) / len(_scores)


def eval_cider(
    predictions: Sequence[str],
    targets: Sequence[Sequence[str]],
    ) -> float:
    """Compute CIDEr score."""
    coco_tokenizer = PTBTokenizer()
    scorer = Cider()
    score, scores = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
    # gts={
    #     str(i): target
    #     for i, target in enumerate(targets)
    # }
    # res={
    #     str(i): [prediction]
    #     for i, prediction in enumerate(predictions)
    # }

    # score, scores = scorer.compute_score(gts=gts, res=res)

    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores.tolist()]
    return score, scores



if __name__ == "__main__":
    # exp_names = ['release/GPE_entire', 'release/GPE_entire', 'release/GPE_RD_entire', 'release/GPE_RD_entire']
    exp_names = ['release/GPE_entire']
    # exp_names = ['sota/llama2_7b_grouphead_public']
    steps = [20000]
    # dataset_names = ['docvqa', 'visualMRC', 'funsd', 'cord', 'sroie', 'newspapers', 'synthtable', 'synthdocs', 'Fetaqa_QA_bbox4_entire', 'ppt_QA_bbox4_entire', '政府网页截图_QA_bbox4_entire']
    dataset_names = ['forms']

    print_list = []
    only_table = False
    
    for exp_name, step in zip(exp_names, steps):
        score_string = f"|{step}|"
        input_dir = f'/workspace3/mosay.zy/finllm_workspace/eval_results/public/predict_{exp_name}_step{step}_2/'
        # input_dir = '/workspace3/mosay.zy/finllm_workspace/eval_results/public/predict_grouphead-base-1w_step20000'
        for dataset_name in dataset_names:
            input_path = os.path.join(input_dir, dataset_name + '.jsonl')
            if not os.path.exists(input_path):
                score_string += "|"
                continue
            input_data = read_jsonl(input_path)
            try:
                refs = []
                all_refs = []
                outs = []
                if not len(input_data):
                    continue
                for sample in input_data:
                    if 'outputs' not in sample:
                        continue
                    out_dict = sample['model_output']['output']
                    questions, answers = sample['inputs'], sample['outputs']
                    if only_table:
                        qas = [[x, y] for x, y in zip(questions, answers) if '\n' in y]
                        if not len(qas):
                            continue
                        questions, answers = zip(*qas)
                        # print(sample['imgname'])
                    if 'all_outputs' in sample:
                        all_answers = sample['all_outputs']
                    else:
                        all_answers = [[item] for item in sample['outputs']]
                    for qidx, question in enumerate(questions):
                        if question in out_dict:
                            # print(question)
                            # print(out_dict[question])
                            outs.append(out_dict[question])
                            refs.append(answers[qidx])
                            all_refs.append(all_answers[qidx])
                # print(calculate_f1(outs, all_refs))
                anls_score = eval_anls(outs, all_refs)
                # print(f"ANLS score for {input_path} is {anls_score}")
                # in_accuracy = eval_in_accuracy(outs, refs)
                # print(f"IN-accuracy for {input_path} is {in_accuracy}")
                # in_accuracy = eval_equal_accuracy(outs, refs)
                # print(f"EQUAL-accuracy for {input_path} is {in_accuracy}")
                cider_score, cider_scores = eval_cider(outs, all_refs)
                # print(f"CIDEr score for {input_path} is {cider_score}")
                anls_score, cider_score = round(anls_score * 100, 2), round(cider_score, 2)
                if dataset_name == 'visualMRC':
                    score_string += f"{anls_score}/{cider_score}|"
                else:
                    score_string += f"{anls_score}|"
            except Exception as E:
                raise E
        print_list.append(score_string)
    print('||' + '|'.join(dataset_names) + '|')
    sep_line = (len(dataset_names) + 1) * ['---']
    print('|' + '|'.join(sep_line) + '|')
    for s in print_list:
        print(s)