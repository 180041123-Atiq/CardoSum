from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def preprocess_function(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": sample["Original_Report"],
            },
            {
                "role": "assistant",
                "content": sample["GPT_Summary"],
            },
        ]
    }

if __name__ == '__main__':
    train_dataset = load_dataset(
        "csv",
        data_files="sft_train.csv",
    )['train']
    val_dataset = load_dataset(
        "csv",
        data_files="val.csv",
    )['train']

    train_dataset = train_dataset.map(preprocess_function, remove_columns=['Original_Report', 'GPT_Summary'])
    val_dataset = val_dataset.map(preprocess_function, remove_columns=['Original_Report', 'GPT_Summary'])
    # print(next(iter(dataset)))

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    config = SFTConfig(
        output_dir='output_sft',
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        save_strategy="steps",
        save_steps=0.2,
        
        eval_on_start=True,
        eval_strategy='steps',
        eval_steps=0.2,

        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,

        # save_total_limit=1,
        
        logging_strategy="steps",
        logging_steps=1,              
        report_to="tensorboard",
    )
    
    trainer = SFTTrainer(
        model='meta-llama/Llama-3.2-3B-Instruct',
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )
    trainer.train()


#     Effective Batch Size	Recommended LR	Safe Range
    # 1	5e-6	3e-6 – 8e-6
    # 2	8e-6	5e-6 – 1e-5
    # 4	1e-5	8e-6 – 1.5e-5
    # 8	1.5e-5	1e-5 – 2e-5
    # 16	2e-5	1.5e-5 – 3e-5
    # 32	3e-5	2e-5 – 4e-5
    # 64	5e-5	3e-5 – 6e-5