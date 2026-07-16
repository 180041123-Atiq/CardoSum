import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from datasets import load_dataset
import os
from tqdm import tqdm

from utils import conv_icl2

class LLMInference:

    def __init__(
        self,
        model_name,
        adapter_path=None,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype='auto',
            device_map='auto'
        )

        if adapter_path is not None:
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    ):
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt},
            ]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise Exception('Unsupported Format for prompt')

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
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


if __name__ == "__main__":

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    ADAPTER = "output_awgrpo/checkpoint-200"
    # ADAPTER = None

    llm = LLMInference(
        model_name=MODEL_NAME,
        adapter_path=ADAPTER,
    )

    test_dataset = load_dataset(
        "csv",
        data_files="test.csv",
    )['train']
    train_dataset = load_dataset(
        "csv",
        data_files="train.csv",
    )['train']
    exp1,exp2 = train_dataset[4], train_dataset[6]

    output_dir = 'output_awgrpo/awgrpo_outputs_200'
    os.makedirs(output_dir, exist_ok=True)
    sample_cnt = 0
    
    for sample in tqdm(test_dataset, total=len(test_dataset), desc=f'Inferencing {MODEL_NAME}'):
        # print(conv_icl2(exp1,exp2,sample))
        # break
        response = llm.generate(
            # prompt = conv_icl2(exp1,exp2,sample),
            prompt = sample['Original_Report'],
            max_new_tokens=430,
        )
    
        with open(f'{output_dir}/{sample_cnt}.txt','w') as f:
            f.write(response)
        sample_cnt += 1