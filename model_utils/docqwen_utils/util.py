# -*- encoding: utf-8 -*-
import json
import tqdm
import os
import base64
import requests
import cv2
import shutil
import numpy as np
import tqdm


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def dump_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_example(example: dict) -> dict:
    context = f"{example.get('instruction', '')}\n"
    if example.get("input"):
        context += f"{example['input']}\n"
    target = example["output"]
    data = {"context": context, "target": target, "imgname": example.get('imgname', ''), "instruct_type": example.get('instruct_type', 'ie')}
    context_bbox = example.get('instruct_bbox')
    if context_bbox is not None:
        context_bbox = json.dumps(context_bbox.tolist())
        data.update({"context_bbox": context_bbox})
    return data


def load_jsonl(filepath, json_format=True):
    with open(filepath) as f:
        lines = [x.strip() for x in f.readlines()]
    if json_format:
        lines = [json.loads(x.strip()) for x in lines]
    return lines


def dump_jsonl(data, save_path, format_func=False):
    with open(save_path, 'w') as f:
        for idx, example in tqdm.tqdm(enumerate(data), desc="formatting.."):
            if format_func:
                f.write(json.dumps(format_example(example), ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')


def rotate_image(imageRaw, data):
    data = data['resultMap']
    angle = data.get('angle_cls', 0)
    angle_score = data.get('angle_score', 0.)
    if angle != 0 and angle_score > 0.7:
        image_np = base64_to_numpy(imageRaw)
        h, w = image_np.shape[:2]
        if angle == 1:
            image_np = np.ascontiguousarray(np.rot90(np.rot90(np.rot90(image_np))))
        elif angle == 2:
            image_np = np.ascontiguousarray(np.rot90(np.rot90(image_np)))
        elif angle == 3:
            image_np = np.ascontiguousarray(np.rot90(image_np))
        imageRaw = numpy_to_base64(image_np)
    return imageRaw


def base64_to_numpy(image_string):
    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def numpy_to_base64(img_np):
    retval, buffer = cv2.imencode('.jpg', img_np)
    pic_str = base64.b64encode(buffer).decode("utf-8")
    return pic_str


def request_server(data, url):
    body = {
        "features": {'imageRaw': data, 'Angle': 1}
    }
    headers = {
        "Content-Type": "application/json",
        "MPS-app-name": "test",
        "MPS-http-version": "1.0",
        "MPS-trace-id": "trace_id",
    }
    response = requests.request("POST", url, data=json.dumps(body), headers=headers)
    assert response.status_code == 200, f"{url} failed! response={response}"
    res = json.loads(response.text)
    if "resultMap" in res and "resultMap" in res["resultMap"]:
        res["resultMap"]["resultMap"] = json.loads(res["resultMap"]["resultMap"])
    return res
