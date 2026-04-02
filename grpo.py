import pandas as pd
from datasets import Dataset
from trl import GRPOConfig
from csr_trainer import CSRTrainer
from peft import LoraConfig

from csr_reward_fn import reward_fn

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

    config = GRPOConfig(
        output_dir="./debug_grpo",
    
        # 🔹 CORE (must set)
        num_generations=4,                 # VERY IMPORTANT (G > 1)
        per_device_train_batch_size=4,     # small batch
        max_completion_length=64,          # keep fast
    
        # 🔹 TRAIN CONTROL
        max_steps=1,                       # only 1 step
        learning_rate=1e-6,
    
        # 🔹 SAMPLING (to ensure diversity)
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    
        # 🔹 LOGGING
        logging_steps=1,
        report_to="none",
    
        # 🔹 STABILITY
        bf16=True,                         # if GPU supports
        gradient_checkpointing=False,      # disable for debugging
    
        # 🔹 IMPORTANT (for your case)
        num_iterations=1,
        scale_rewards="group",             # default is fine

        # generation_kwargs={
        #     "num_return_sequences": 4,
        #     "do_sample": True,
        # }
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # important for LLaMA
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    trainer = CSRTrainer(
        model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset_dict['train'],
        peft_config=peft_config,
    )
    trainer.train()
    