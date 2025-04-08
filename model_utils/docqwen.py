from .docqwen_utils.inference import load_model, greedy_search
from tqdm import tqdm



def docqwen_predict(model_path, data_list, start_idx, end_idx, gpu_id=0):
    data_list = data_list[start_idx: end_idx]
    tokenizer, model = load_model(model_path, f"cuda:{gpu_id}")
    out_list = []
    for data in tqdm(data_list):
        out = data
        output_list = greedy_search(model, tokenizer, 128, data)
        assert len(data['inputs']) == len(output_list), f"len(data['inputs']) != len(output_list), please check."
        output_dict = {}
        for q, a in zip(data['inputs'], output_list):
            output_dict[q] = a
        out["model_output"] = output_dict
        out_list.append(out)
    return out_list