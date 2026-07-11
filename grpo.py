import pandas as pd
from datasets import Dataset
from trl import GRPOConfig
from csr_trainer import CSRTrainer
from peft import LoraConfig
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

from csr_reward_fn import calc_similarity_reward, calc_readability_reward, calc_concise_reward

def get_dataset_dict(paths):
    dataset_dict = {}
    for k,v in paths.items():
        df = pd.read_csv(v)
        
        df = df.rename(columns={
            "Original_Report": "prompt",
            "GPT_Summary": "reference"
        })

        df['prompt'] = df['prompt'].apply(lambda x: [{"role": "user", "content": x}])

        dataset_dict[k] = Dataset.from_pandas(df)

    return dataset_dict

if __name__ == '__main__':
 
    paths = {
        'train': 'grpo_train.csv',
        'val': 'val.csv',
    }
    dataset_dict = get_dataset_dict(paths)
    
    config = GRPOConfig(
        output_dir="./output_grpo",
    
        num_generations=4,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        max_completion_length=640, # since max prompt can be 707, compression is 1.12, so ref should be 707/1.12 = 640

        # eval_on_start=True,
        # eval_strategy="steps",
        # eval_steps=0.2,
    
        save_strategy="steps",
        save_steps=0.2,
    
        # reward_weights=[-1.0, -1.0, -1.0],
        # multi_objective_aggregation='AWGRPO',
        logging_strategy='steps',
        logging_steps=1,              # log every N steps
        report_to="tensorboard", 
    )

    # peft_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    # )

    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    lora_path = 'output_sft/checkpoint-2250'
    
    processing_class = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype='auto',
        device_map='auto'
    )

    model = PeftModel.from_pretrained(
        model,
        lora_path,
        device_map="auto",
        is_trainable=True,
    )
    
    trainer = CSRTrainer(
        model=model,
        processing_class=processing_class,
        reward_funcs=[calc_similarity_reward, calc_readability_reward, calc_concise_reward],
        args=config,
        train_dataset=dataset_dict['train'],
        # eval_dataset=dataset_dict['val'],
    )

    trainer.train()
    