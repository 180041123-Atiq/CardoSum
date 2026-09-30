import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from datasets import load_dataset
import os
from tqdm import tqdm
from evaluate import load as eval_load

from utils import conv_icl2

class LLMInference:

    def __init__(
        self,
        model_name,
        adapter_path=None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype='auto' if 'medalpaca' not in model_name else torch.float16,
            device_map='auto'
        )

        if adapter_path is not None:
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )

        self.model.eval()

        if 'medalpaca' in model_name:
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        instruct=False,
    ):
        if isinstance(prompt, str): 
            if instruct:
                messages = [
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise Exception('Unsupported Format for prompt')

        model_max_length = self.model.config.max_position_embeddings
        max_input_length = model_max_length - max_new_tokens

        # print(f'max_input_length is : {max_input_length}')
        # print(f'model_max_length is : {model_max_length}')
        # print(f'max_new_tokens is : {max_new_tokens}')

        if instruct:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                max_length=max_input_length,
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

def eval_score(preds, refs, file_name):
    rouge = eval_load("rouge")
    bertscore = eval_load("bertscore")

    # Compute metrics
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    print({
        "ROUGE-L": rouge_res["rougeL"],
        "BERTScore-F1": sum(bert_res["f1"]) / len(bert_res["f1"]),
    })

    with open(file_name,'w') as f:
        f.write(f"ROUGE-L: {rouge_res['rougeL']}\n")
        f.write(f"BERTScore-F1: {sum(bert_res['f1']) / len(bert_res['f1'])}")

def evaluation(pred_dir):
    preds, refs = [], []
    test_dataset = load_dataset('csv',data_files='test.csv')['train']
    # pred_dir = 'output_awgrpo/awgrpo_outputs_200'
    for ii,sample in enumerate(tqdm(test_dataset,total=len(test_dataset),desc='Converting samples for eval')):
        # print(sample)
        # break
        pred, ref = '', sample['GPT_Summary']
        with open(f'{pred_dir}/{ii}.txt', 'r') as f:
            pred = f.read()
        # print()
        # print(pred)
        # print()
        # print(ref)
        # break
        preds.append(pred)
        refs.append(ref)
        
    eval_score(preds,refs,f'results/{pred_dir}_eval_score.txt')

if __name__ == "__main__":

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    # MODEL_NAME = "medalpaca/medalpaca-7b"
    # MODEL_NAME = "epfl-llm/meditron-7b"
    ADAPTER = "output_HVO/checkpoint-1000"
    # ADAPTER = None

    llm = LLMInference(
        model_name=MODEL_NAME,
        adapter_path=ADAPTER,
    )

    test_dataset = load_dataset(
        "csv",
        data_files="test.csv",
    )['train']
    # train_dataset = load_dataset(
    #     "csv",
    #     data_files="train.csv",
    # )['train']
    # exp1,exp2 = train_dataset[4], train_dataset[6]

    output_dir = 'gen_outputs_HVO'
    os.makedirs(output_dir, exist_ok=True)
    sample_cnt = 0
    
    for sample in tqdm(test_dataset, total=len(test_dataset), desc=f'Inferencing {MODEL_NAME}'):
        # print(conv_icl2(exp1,exp2,sample))
        # break
        response = llm.generate(
            # prompt = conv_icl2(exp1,exp2,sample),
            prompt = sample['Original_Report'],
            max_new_tokens=640,
            instruct=True,
        )
    
        with open(f'{output_dir}/{sample_cnt}.txt','w') as f:
            f.write(response)
        sample_cnt += 1

    # Need to check this
    evaluation(output_dir)