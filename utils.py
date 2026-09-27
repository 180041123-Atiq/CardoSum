import zipfile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def conv_icl2(exp1, exp2, exp3):
    return [
        {
            "role": "system",
            "content": (
                "You are a clinical communication expert who "
                "explains echocardiography reports in clear, accurate, "
                "patient-friendly language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Below are two examples showing how to summarize an "
                "echocardiography report into an easy-to-understand explanation.\n\n"

                "### Example 1\n"
                "Input:\n"
                + exp1['Original_Report'].split(
                    "Please explain the following Echocardiography report."
                )[1]
                + "\n\n"
                "Output:\n"
                + exp1['GPT_Summary']
                + "\n\n"

                "### Example 2\n"
                "Input:\n"
                + exp2['Original_Report'].split(
                    "Please explain the following Echocardiography report."
                )[1]
                + "\n\n"
                "Output:\n"
                + exp2['GPT_Summary']
                + "\n\n"

                "Now explain the following report in the same way.\n\n"

                "Input:\n"
                + exp3['Original_Report'].split(
                    "Please explain the following Echocardiography report."
                )[1]
                + "\n\n"
                "Output:"
            ),
        },
    ]

def unzip_file(source_file, out_dir):
    with zipfile.ZipFile(source_file, 'r') as zip_ref:
        zip_ref.extractall(out_dir)


def tensorboard_event_reader(file_path):
    
    ea = EventAccumulator(file_path)
    ea.Reload()
    
    # print(ea.Tags())
    # print(ea.Scalars('train/reward'))
    # eval_loss = ea.Scalars("eval/loss")
    # print(eval_loss)

    reward_events = ea.Scalars("train/reward")
    calc_similarity_events = ea.Scalars('train/rewards/calc_similarity_reward/mean')
    calc_concise_events = ea.Scalars('train/rewards/calc_concise_reward/mean')
    calc_readability_events = ea.Scalars('train/rewards/calc_readability_reward/mean')
    for event in reward_events:
        if event.step in [200, 400, 600, 800, 1000]:
            print(f"Step {event.step}: Reward = {event.value:.4f}")
    for event in calc_similarity_events:
        if event.step in [200, 400, 600, 800, 1000]:
            print(f"Step {event.step}: Similarity Reward Mean = {event.value:.4f}")
    for event in calc_concise_events:
        if event.step in [200, 400, 600, 800, 1000]:
            print(f"Step {event.step}: Concise Reward Mean = {event.value:.4f}")
    for event in calc_readability_events:
        if event.step in [200, 400, 600, 800, 1000]:
            print(f"Step {event.step}: Redability Reward Mean = {event.value:.4f}")
    
    
def correcting_samples():
    
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    COLUMN_NAME = "Original_Report" ## or "GPT_SUMMARY"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_dataset("csv",data_files=["train.csv"],)['train']

    text = dataset[COLUMN_NAME][7434] 

    tokens = tokenizer(text,add_special_tokens=True,truncation=False,)["input_ids"]

    print(text)
    print(len(tokens))

def token_count():

    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    # COLUMN_NAME = "Original_Report" ## or"GPT_Summary"
    COLUMN_NAME = "GPT_Summary" ## "Original_Report"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset(
        "csv",
        data_files=["train.csv","val.csv","test.csv"],
    )['train']
    
    lengths = []

    excluded_list = [4405, 895, 9769, 244, 2464, 2299, 2814, 3310, 3905, 8693, 7889, 337, 4141, 310, 7434]
    
    for ii,sample in enumerate(dataset):
        # text = sample["GPT_Summary"]          # Change "text" to your column name
        text = sample[COLUMN_NAME]          # Change "text" to your column name
        # print(text)
        # break
        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        if ii in excluded_list:
            continue
        
        lengths.append(len(tokens))
        # if ii==4: print(ii, len(tokens))
        # if ii==6: print(ii, len(tokens))
        # if len(tokens) == 144: print(text,ii)
        # if ii == 7434:
        #     print(text)
        # if len(tokens) > 434: 
        #     print()
        #     print(ii,len(tokens))
        #     print()
    
    lengths = np.array(lengths)
    
    # print(f"Number of samples : {len(lengths)}")
    # print(f"Min tokens        : {lengths.min()}")
    # print(f"Max tokens        : {lengths.max()}")
    # print(f"Mean tokens       : {lengths.mean():.2f}")
    # print(f"Median tokens     : {np.median(lengths):.2f}")
    # print(f"Std deviation     : {lengths.std():.2f}")
    
    # print(f"95th percentile   : {np.percentile(lengths,95):.2f}")
    # print(f"99th percentile   : {np.percentile(lengths,99):.2f}")

    statistics = {
        "number_of_samples": int(len(lengths)),
        "min_tokens": int(lengths.min()),
        "max_tokens": int(lengths.max()),
        "mean_tokens": round(float(lengths.mean()), 2),
        "median_tokens": round(float(np.median(lengths)), 2),
        "std_deviation": round(float(lengths.std()), 2),
        "95th_percentile": round(float(np.percentile(lengths, 95)), 2),
        "99th_percentile": round(float(np.percentile(lengths, 99)), 2)
    }

    output_file = f"{COLUMN_NAME}_token_statistics.json"

    with open(f'results/{output_file}', "w", encoding="utf-8") as f:
        json.dump(statistics,f,indent=4,ensure_ascii=False)

    print(f'Statistics saved at {output_file}')

if __name__ == '__main__':
    token_count()
    # correcting_samples()
    # tensorboard_event_reader('output_awgrpo/runs/Jul15_21-37-05_6ef3283a98b8/events.out.tfevents.1784151425.6ef3283a98b8.8212.0')
    # pass
    # # print(gen_layout_prompt(106))
    # # plot_graph()
    # unzip_file('checkpoint-720.zip','HVO')