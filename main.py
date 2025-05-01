from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import runpod
import torch

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained(""rizkyramadhana26/t5-pii-masking-ai4privacy", device_map="auto", trust_remote_code=True)
    
def handler(job):
    text = job['input']['text']
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model(
            **inputs,
            temperature=0.1,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

runpod.serverless.start({"handler": handler})