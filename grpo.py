import pandas as pd
from datasets import Dataset
from trl import GRPOConfig
from csr_trainer import CSRTrainer
from peft import LoraConfig
import os

from csr_reward_fn import calc_similarity_reward, calc_readability_reward, calc_concise_reward

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

alpaca_prompt_grpo = """Below is an echocardiography report. Summarize it in clear, patient-friendly language.

### Report:
{}

### Summary:"""

def get_dataset_dict(paths):
    dataset_dict = {}
    for k,v in paths.items():
        df = pd.read_csv(v)
        
        df = df.rename(columns={
            "Original_Report": "prompt",
            "GPT_Summary": "reference"
        })
        
        initial_prompt = 'Please explain the following Echocardiography report. '
        for ii in range(df.shape[0]):
            if initial_prompt in df.loc[ii,'prompt']:
                df_str = df.loc[ii,'prompt']
                df.loc[ii,'prompt'] = df_str.split(initial_prompt)[1]

        df['prompt'] = df['prompt'].apply(lambda x: [{"role": "user", "content": alpaca_prompt_grpo.format(x)}])

        dataset_dict[k] = Dataset.from_pandas(df)

    return dataset_dict

    

if __name__ == '__main__':

    paths = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv'
    }
    dataset_dict = get_dataset_dict(paths)
    # print(len(dataset_dict['train']))
    # print(len(dataset_dict['val']))
    # print(len(dataset_dict['test']))
    # exit()
    
    config = GRPOConfig(
        output_dir="./HVO",
    
        num_generations=4,
        per_device_train_batch_size=16,
        
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        
        max_completion_length=640, # since max prompt can be 707, compression is 1.12, so ref should be 707/1.12 = 640
        num_iterations=1,
    
        # eval_strategy="steps",
        # eval_steps=0.1,
    
        save_strategy="steps",
        save_steps=0.2,
        save_total_limit=1,             
        load_best_model_at_end=False,
    
        save_only_model=True,            # avoids optimizer states
    
        reward_weights=[-1.0, -1.0, -1.0],
        multi_objective_aggregation='HVO',

        logging_steps=0.002,              # log every N steps
        report_to="tensorboard", 
    )

    # simple finetuning
    # peft_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # for hard generation task
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    trainer = CSRTrainer(
        model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        reward_funcs=[calc_similarity_reward, calc_readability_reward, calc_concise_reward],
        args=config,
        train_dataset=dataset_dict['train'],
        # eval_dataset=dataset_dict['val'].select(range(100)), # just for sanity checking
        peft_config=peft_config,
    )
    trainer.train()
    