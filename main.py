from peft import PeftModel, PeftConfig
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import runpod
import torch

t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("rizkyramadhana26/t5-pii-masking-ai4privacy", device_map="auto", trust_remote_code=True)
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto", trust_remote_code=True)
llama_model.load_adapter("rizkyramadhana26/llama-3.1-pii-masking-ai4privacy-v3")
classifier = pipeline("ner", model="yeana/my_extractive_app")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def handler(job):
    text = job['input']['text']
    model = job['input']['model']
    if model == 't5':
        inputs = t5_tokenizer(text, return_tensors="pt").to(t5_model.device)
        with torch.no_grad():
            output = t5_model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9
            )
        generated_text = t5_tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    elif model == 'llama':
        text = f"PLEASE REDACT ALL PERSONALLY IDENTIFIABLE INFORMATION FROM TEXT BELOW\n\n==========\n{text}\n\n==========\nYOUR ANSWER:\n"
        inputs = llama_tokenizer(text, return_tensors="pt").to(llama_model.device)
        with torch.no_grad():
            output = llama_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id = llama_tokenizer.eos_token_id
            )
        generated_text = llama_tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text
    elif model == 'bert':
        labels = classifier(text)
        labels = sorted(labels, key=lambda x: x['start'])
        merged = []
        for p in labels:
            start, end = p['start'], p['end']
            ent = p['entity']
            core = ent.split('-', 1)[-1] if '-' in ent else ent

            if ent.startswith('B-') or not merged:
                merged.append({'start': start, 'end': end, 'entity': core})
            else:
                last = merged[-1]
                if last['entity'] == core and start <= last['end']:
                    last['end'] = max(last['end'], end)
                else:
                    merged.append({'start': start, 'end': end, 'entity': core})
        for span in sorted(merged, key=lambda x: x['start'], reverse=True):
            text = text[:span['start']] + f"[{span['entity']}]" + text[span['end']:]
        return text
    elif model == 'presidio':
        analyzer_results = analyzer.analyze(text=text, language='en')
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators={"DEFAULT": OperatorConfig("replace")})
        return anonymized_text.text
    else:
        return 'Error!'
    

runpod.serverless.start({"handler": handler})