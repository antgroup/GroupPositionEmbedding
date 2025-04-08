import torch
import json
import tqdm
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import base64
from PIL import Image
import io
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def qwen2vl_predict(model_path, data_list, image_root, start_idx, end_idx, gpu_id=0):
    data_list = data_list[start_idx: end_idx]
    device = torch.device(f"cuda:{gpu_id}")  
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    out_list = []
    for data in data_list:
        out = data
        img = f"{os.path.splitext(data['imgname'])[0]}.jpg"
        imgpath = os.path.join(image_root, img)
        if not os.path.exists(imgpath):
            img = f"{os.path.splitext(data['imgname'])[0]}.png"
            imgpath = os.path.join(image_root, img)
        output_dict = {}
        for q in data['inputs']:
            prompt = f"{''.join(data['instruction'])}，{q}对应的值是什么？"
            res = chat(model, processor, imgpath, prompt)
            output_dict[q] = res
        out["model_output"] = output_dict
        out_list.append(out)
    return out_list
    

def chat(model, processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # 将所有张量移动到同一个设备（例如 cuda:0）
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    res = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return res[0]
