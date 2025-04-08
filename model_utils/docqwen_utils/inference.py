# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import requests
import math
import logging
import cv2
import base64
from collections import defaultdict
from transformers.cache_utils import Cache, DynamicCache
import re
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from .bbox_processor import BboxProcessor, BBoxUtil
from .ocr_processor import OCRProcessor
import traceback
import copy
import tqdm


bbu = BBoxUtil(with_subfield=False, with_sop=False)
bbox_processor = BboxProcessor()
ocr_pro = OCRProcessor(bbox_to_horizon=False)


def load_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_instruction(instr, ocr=None):
    if ocr is not None and len(ocr):
        ocr_text, ocr_bbox, ocr_char_bbox = ocr_pro.preprocess(ocr)
        ocr_bbox = ocr_pro.box_norm(ocr_bbox)
        ocr_text = [instr] + ocr_text
        instruct_bbox = np.pad(ocr_bbox, ((1, 1), (0, 0)), constant_values=(0, 0))
    else:
        ocr_text = [instr]
        instruct_bbox = [[0, 0, 0, 0]]
    
    ocr_text, instruct_bbox = zip(*[(x, y) for x, y in zip(ocr_text, instruct_bbox) if len(x)]) # filter empty text
    ocr_text = list(ocr_text) if isinstance(ocr_text, tuple) else ocr_text
    instruct_bbox = list(instruct_bbox) if isinstance(instruct_bbox, tuple) else instruct_bbox
    assert len(instruct_bbox) == len(ocr_text), f"instruct_bbox length is {len(instruct_bbox)}, while ocr_text length is {len(ocr_text)}"
    return {'instruction': ocr_text, 'instruct_bbox': instruct_bbox}


def data_preprocess(
    sample,
    tokenizer,
    system_text="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    user_prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
    user_prompt_prefix="<|im_start|>user\n",
    user_prompt_suffix="<|im_end|>\n<|im_start|>assistant\n",
    assistant_prompt="{answer}<|im_end|>\n",
    max_seq_length=2048
    ):
    if 'instruction' in sample and 'instruct_bbox' in sample:
        instruction = sample['instruction'] if isinstance(sample['instruction'], list) else [sample['instruction']]
        instruction_bbox = sample.get('instruct_bbox', [[0, 0, 0, 0]] * len(instruction))
        instruction_ids = [tokenizer.encode(itm, add_special_tokens=False) for itm in instruction]
        instruction_bbox = [len(inst) * [bbox] for inst, bbox in zip(instruction_ids, instruction_bbox)] if len(instruction_ids) and len(instruction_bbox) else []
        instruction_bbox = [x_ for x in instruction_bbox for x_ in x]
        instruction_ids = [x_ for x in instruction_ids for x_ in x]
    else:
        instruction_ids, instruction_bbox = [], []

    # 纯文本数据用input和output
    if 'input' in sample:
        sample['inputs'] = [sample.pop('input')]
    
    querys = sample['inputs']
    input_ids = tokenizer(system_text).input_ids

    # add first question id to input_ids, others to q_ids_list
    q_ids_list = []
    for idx, q in enumerate(querys):
        if idx == 0:
            query_id = tokenizer.encode(q, add_special_tokens=False)
            user_prompt_prefix_id = tokenizer.encode(user_prompt_prefix, add_special_tokens=False)
            user_prompt_suffix_id = tokenizer.encode(user_prompt_suffix, add_special_tokens=False)
            q_ids = user_prompt_prefix_id + instruction_ids + query_id + user_prompt_suffix_id
            instruction_bbox = [[0, 0, 0, 0]] * (len(input_ids) + len(user_prompt_prefix_id)) + instruction_bbox
        else:
            q_ids = tokenizer(user_prompt.format(query=q)).input_ids
        if idx == 0:
            input_ids += q_ids
        else:
            q_ids_list.append(q_ids)
    data = {
        'bbox': instruction_bbox,
        'input_ids': input_ids,
        'q_ids_list': q_ids_list
    }
    return data


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True)
    print(f"loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
            # torch_dtype=torch.bfloat16 
        ).half().to(device)
    model.config.use_cache = False
    print("Restored ckpt from {}".format(model_path))
    model = model.eval()
    return tokenizer, model


def greedy_search(
    model,
    tokenizer,
    max_new_tokens,
    example
    ):
    with torch.no_grad():
        device = model.device
        ocr_text = example['instruction']
        processed_data = data_preprocess(example, tokenizer)
        input_ids, prompts_bbox, q_ids_list = processed_data['input_ids'], processed_data['bbox'], processed_data['q_ids_list']
        idx = 0
        answer = ''
        pred_inputs, pred_outputs = [], []
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.update(max_new_tokens=max_new_tokens)
        model_kwargs = generation_config.to_dict()
        eos_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token, '<|im_end|>'])

        forward_cnt = 0
        scores = []
        while idx < len(q_ids_list) + 1:
            score = []

            if idx == 0:
                round_prompt_id = input_ids
                length = len(round_prompt_id)
            else:
                q_ids = q_ids_list[idx-1]
                round_prompt_id = round_input_id[0].tolist() + q_ids
                # update attention mask
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], len(q_ids)))], dim=-1
                    )
            
            pad_num = len(round_prompt_id) - len(prompts_bbox)
            round_prompt_bbox = np.pad(prompts_bbox, ((0, pad_num), (0, 0)), constant_values=(0, 0))
            
            round_prompt_id = torch.tensor(np.array([round_prompt_id]))
            round_prompt_bbox = torch.tensor(np.array([round_prompt_bbox]))
            
            round_input_id = round_prompt_id.to(device).to(dtype=torch.long)
            # attention_mask = torch.ones(1, length).to(device).to(dtype=torch.long)
            model_kwargs['bbox'] = round_prompt_bbox.to(device).to(dtype=torch.long)
            model_kwargs['max_new_tokens'] = max_new_tokens
            model_kwargs['temperature'] = 0
            model_kwargs['cache_position'] = torch.tensor([0])
            
            forward_cnt = 0
            while True:
                model_inputs = model.prepare_inputs_for_generation(round_input_id, **model_kwargs)
                # forward pass to get next token
                t0 = time.time()
                outputs = model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                forward_cnt += 1

                next_token_logits = outputs.logits[:, -1, :]
                score.append(round(next_token_logits.softmax(-1).max().item(), 2))

                # argmax
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                # update generated ids, model inputs, and length for next step 
                # torch.Size([1, 546]) -> torch.Size([1, 547])
                round_input_id = torch.cat([round_input_id, next_tokens[:, None]], dim=-1)
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=False,
                )
                if next_tokens in eos_id or forward_cnt > max_new_tokens:
                    break
            scores.append(score)
            out_token = tokenizer.decode(round_input_id[0][len(round_prompt_id[0]): -1])
            input_token = example['inputs'][idx]
            pred_inputs.append(input_token)
            pred_outputs.append(out_token)

            idx += 1
            answer += (input_token + out_token + '\n')
        # return answer, '|'.join(ocr_text), pred_inputs, pred_outputs, scores
        return pred_outputs


def distribute_data_across_gpus(data, num_gpus):
    """Distribute data evenly across GPUs."""
    np.random.shuffle(data)
    batch_size = len(data) // num_gpus
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_gpus)]
    # Handle the remainder
    remainder = len(data) % num_gpus
    if remainder:
        for i in range(remainder):
            batches[i].append(data[-(i+1)])
    return batches


def evaluate_model_on_gpu(gpu_id, data_batch, model_path, max_new_tokens, results):
    """Evaluate the model on a specific GPU."""
    tokenizer, model = load_model(model_path, f"cuda:{gpu_id}")
    for example in tqdm.tqdm(data_batch):
        example["model_output"] = greedy_search(
                model,
                tokenizer,
                max_new_tokens,
                example)
        results.append(example)


def evaluate_model(data, model_path, outpath, max_new_tokens=16, gpu_num=8):
    """Evaluate the model on different tasks using multiple GPUs."""
    # Distribute data across GPUs
    data_batches = distribute_data_across_gpus(data, gpu_num)
    manager = Manager()
    results = manager.list()
    processes = []

    if gpu_num <= 1:
        evaluate_model_on_gpu(0, data_batches[0], model_path, max_new_tokens, results)
    else:
        for gpu_id, data_batch in enumerate(data_batches):
            p = Process(target=evaluate_model_on_gpu, args=(gpu_id, data_batch, model_path, max_new_tokens, results))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    dump_jsonl(results, outpath)


def inference_example(model_path, max_new_tokens, ocr, inputs, instruction='Answer the question according to the following infomation.'):
    example = get_instruction(instruction, ocr)
    example['inputs'] = inputs
    results = []
    evaluate_model_on_gpu(0, [example], model_path, max_new_tokens, results)
    return results


if __name__ == '__main__':
    model_path = '/workspace3/mosay.zy/finllm_workspace/model_output/qwen2_mmtask/release/0207/step-15000'
    max_new_tokens = 10
    ocr = [
        {'content': 'bank name: mybank', 'bndbox': [[55, 78], [670, 74], [670, 205], [55, 208]]}, 
        {'content': '邕e登-我的不动产', 'bndbox': [[230, 527], [1098, 527], [1098, 638], [230, 638]]}
    ]
    inputs = ['what is the bank name?']
    instruction = 'Answer the question according to the following infomation.'
    results = inference_example(model_path, max_new_tokens, ocr, inputs, instruction)
    print(results)


    # data_file = '/workspace3/ocr_data/ocr_data_collect/public_datasets/LayoutLLM_data/LayoutLLM_data/eval_data/VIE_CORD.jsonl'
    # outpath = 'output.jsonl'
    # data = load_data(data_file)
    # evaluate_model(data, model_path, outpath, max_new_tokens=16, gpu_num=8)