import zipfile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

def unzip_file(source_file, out_dir):
    with zipfile.ZipFile(source_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)


def tensorboard_event_reader(file_path):
    
    ea = EventAccumulator(file_path)
    ea.Reload()
    
    eval_loss = ea.Scalars("eval/loss")
    print(eval_loss)

def token_count():

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset(
        "csv",
        data_files="test.csv",
    )['train']
    
    lengths = []
    
    for sample in dataset:
        text = sample["GPT_Summary"]          # Change "text" to your column name
        # print(text)
        # break
        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    
        lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    print(f"Number of samples : {len(lengths)}")
    print(f"Min tokens        : {lengths.min()}")
    print(f"Max tokens        : {lengths.max()}")
    print(f"Mean tokens       : {lengths.mean():.2f}")
    print(f"Median tokens     : {np.median(lengths):.2f}")
    print(f"Std deviation     : {lengths.std():.2f}")
    
    print(f"95th percentile   : {np.percentile(lengths,95):.2f}")
    print(f"99th percentile   : {np.percentile(lengths,99):.2f}")

if __name__ == '__main__':
    token_count()
    # tensorboard_event_reader('runs\Jul10_19-45-30_c51a494ff5d9\events.out.tfevents.1783712730.c51a494ff5d9.3314.0')
    # pass
    # # print(gen_layout_prompt(106))
    # # plot_graph()
    # unzip_file('checkpoint-720.zip','HVO')