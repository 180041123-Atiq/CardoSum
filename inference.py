import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from datasets import load_dataset
from evaluate import load as eval_load

class LLMInference:

    def __init__(
        self,
        model_name,
        adapter_path=None,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
        )

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

        messages = [
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        generated = outputs[0][input_ids.shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )



def eval_score(preds, refs, file_name):
    rouge = eval_load("rouge")
    bertscore = eval_load("bertscore")

    # Suppose you have one prediction and one ground truth
    # pred = "Your echocardiography report shows mild mitral regurgitation..."
    # ref = "The echocardiogram reveals mild mitral regurgitation with normal LV function."

    # # Wrap them in lists
    # preds = [pred]
    # refs = [ref]

    # Compute metrics
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    print({
        "ROUGE-L": rouge_res["rougeL"],
        "BERTScore-F1": sum(bert_res["f1"]) / len(bert_res["f1"]),
    })

    with open(os.path.join(OUTPUT_DIR,file_name),'w') as f:
        f.write(f"ROUGE-L: {rouge_res['rougeL']}\n")
        f.write(f"BERTScore-F1: {sum(bert_res['f1']) / len(bert_res['f1'])}")

if __name__ == "__main__":

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    ADAPTER = "./lora_checkpoint"

    llm = LLMInference(
        model_name=MODEL_NAME,
        adapter_path=ADAPTER,
    )

    test_dataset = load_dataset(
        "csv",
        data_files="test.csv",
    )['train']

    for sample in test_dataset:
        response = llm.generate(
            sample['Original_Report'],
            max_new_tokens=430,
        )
    
        print(response)