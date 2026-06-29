import torch
import re

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def generate(prompt, model, tokenizer):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    out = model.generate(inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
 
def exact_match(pred, gt):
    # to measure strict prescription correctness
    return normalize(pred) == normalize(gt)


## token level f1 score

def token_f1(pred, gt):
    pred_tokens = normalize(pred).split()
    gt_tokens = normalize(gt).split()

    common_tokens = set(pred_tokens) & set(gt_tokens)

    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


