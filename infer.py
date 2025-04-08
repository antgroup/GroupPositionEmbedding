import os
import json
import concurrent.futures
import multiprocessing
from config import dataset_list
from model_utils.qwen2vl import qwen2vl_predict
from model_utils.qwen2 import qwen2_predict
from model_utils.llama2 import llama2_predict
from model_utils.docqwen import docqwen_predict
import argparse


def dump_jsonl(data, file_path):
    lines = [json.dumps(x, ensure_ascii=False) for x in data]
    with open(file_path, 'w', encoding='utf8') as fout:
        fout.write('\n'.join(lines))


def infer_dataset(args):
    dataset_list = args['dataset_list']
    model_type = args['model_type']
    image_type = args['image_type']
    gpu_num = args['gpu_num']
    infer_step = args['infer_step']
    model_paths = args['model_paths']

    if not multiprocessing.get_start_method(allow_none=True):
        multiprocessing.set_start_method('spawn')

    for dataset_name, dataset_file in dataset_list.items():
        print(f'Start infer {dataset_name}...')
        out_path = f"./results/{image_type}/{model_type}/{dataset_name}.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        image_root = os.path.join(os.path.dirname(dataset_file), image_type) if model_type == 'qwen2vl' else None

        with open(dataset_file, 'r') as f:
            data_list = [json.loads(line) for line in f]

        res = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=gpu_num) as executor:
            futures = []
            num_steps = min(len(data_list), infer_step) if infer_step > 0 else len(data_list)
            step_size = num_steps // gpu_num

            for j in range(gpu_num):
                start_idx = j * step_size
                end_idx = min(start_idx + step_size, num_steps)

                predict_func = None
                if model_type == 'qwen2vl':
                    predict_func = qwen2vl_predict
                elif model_type == 'qwen2':
                    predict_func = qwen2_predict
                elif model_type == 'llama2':
                    predict_func = llama2_predict
                elif model_type == 'docqwen':
                    predict_func = docqwen_predict
                if model_type == 'qwen2vl':
                    futures.append(executor.submit(predict_func, model_paths[model_type], data_list, image_root, start_idx, end_idx, j))
                else:
                    futures.append(executor.submit(predict_func, model_paths[model_type], data_list, start_idx, end_idx, j))

            for future in concurrent.futures.as_completed(futures):
                res.extend(future.result())

        dump_jsonl(res, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument('--model_type', type=str, default='qwen2', choices=['qwen2vl', 'qwen2', 'llama2', 'docqwen'], help="Model type to use")
    parser.add_argument('--image_type', type=str, default='cropped_images', choices=['cropped_images', 'images'], help="Image type to use")
    parser.add_argument('--gpu_num', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--infer_step', type=int, default=0, help="Inference step limit (0 means no limit)")

    args = parser.parse_args()

    try:
        from config import model_paths
    except ImportError:
        print("Error: Please configure the 'model_paths' in a file named 'config.py' in the current directory.")
        print("Example content for config.py:")
        print("""
model_paths = {
    'qwen2vl': '/path/to/qwen2vl',
    'qwen2': '/path/to/qwen2',
    'llama2': '/path/to/llama2',
    'docqwen': '/path/to/docqwen'
}
""")
        exit(1)

    infer_args = {
        'dataset_list': dataset_list,
        'model_type': args.model_type,
        'image_type': args.image_type,
        'gpu_num': args.gpu_num,
        'infer_step': args.infer_step,
        'model_paths': model_paths
    }

    infer_dataset(infer_args)