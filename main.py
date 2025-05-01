from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import runpod
import torch

t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("rizkyramadhana26/t5-pii-masking-ai4privacy", device_map="auto", trust_remote_code=True)
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llama_model.load_adapter("rizkyramadhana26/llama-3.1-pii-masking-ai4privacy-v3")

def handler(job):
    text = job['input']['text']
    model = job['input']['model']
    if model == 't5':
        inputs = t5_tokenizer(text, return_tensors="pt").to(t5_model.device)
        with torch.no_grad():
            output = t5_model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=512
            )
        generated_text = t5_tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    elif model == 'llama':
        inputs = llama_tokenizer(text, return_tensors="pt").to(llama_model.device)
        with torch.no_grad():
            output = llama_model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=512,
                do_sample=True
            )
        generated_text = llama_tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text
    else:
        return 'Error!'
    

runpod.serverless.start({"handler": handler})