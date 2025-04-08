from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def llama2_predict(model_path, data_list, start_idx, end_idx, gpu_id=0):
    data_list = data_list[start_idx: end_idx]
    device = torch.device(f"cuda:{gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

