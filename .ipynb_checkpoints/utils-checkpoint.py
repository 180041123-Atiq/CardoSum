import zipfile
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
# from transformers import AutoTokenizer
# from datasets import load_dataset

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
    # token_count()
    tensorboard_event_reader('output_awgrpo/runs/Jul15_21-37-05_6ef3283a98b8/events.out.tfevents.1784151425.6ef3283a98b8.8212.0')
    # pass
    # # print(gen_layout_prompt(106))
    # # plot_graph()
    # unzip_file('checkpoint-720.zip','HVO')