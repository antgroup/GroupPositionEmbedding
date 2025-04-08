from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def qwen2_predict(model_path, data_list, start_idx, end_idx, gpu_id=0):
    data_list = data_list[start_idx: end_idx]
    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    out_list = []
    for data in data_list:
        out = data
        output_dict = {}
        for q in data['inputs']:
            prompt = f"{''.join(data['instruction'])}，{q}对应的值是什么？请只输出对应的值是什么，不要输出其他内容。如果没有请输出'无',不要输出中间过程。"
            res = chat(model, tokenizer, prompt)
            output_dict[q] = res
        out["model_output"] = output_dict
        out_list.append(out)
    return out_list


def chat(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response