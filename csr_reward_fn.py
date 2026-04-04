from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import numpy as np
import textstat
import pandas as pd
import random

# will try to use the tokenizer the model is using for tokenization
llama_tok_concise_rew = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def hyper_params_csr(paths=None):
    if paths is None:
        paths = [
            'test.csv',
            'val.csv',
            'train.csv'
        ]
    
    data_cr_list, smog_list = [], []

    for csv_path in paths:
        df = pd.read_csv(csv_path)
        
        df = df.rename(columns={
            "Original_Report": "prompt",
            "GPT_Summary": "reference"
        })

        initial_prompt = 'Please explain the following Echocardiography report. '
        for ii in range(df.shape[0]):
            if initial_prompt in df.loc[ii,'prompt']:
                df_str = df.loc[ii,'prompt']
                df.loc[ii,'prompt'] = df_str.split(initial_prompt)[1]

        for ii in range(df.shape[0]):
            len_prompt = len(llama_tok_concise_rew.encode(df.loc[ii,'prompt']))
            ref = df.loc[ii,'reference']
            len_ref = len(llama_tok_concise_rew.encode(ref))
            data_cr_list.append(len_prompt/len_ref)
            smog_list.append(textstat.smog_index(ref))

    smog_array = np.array(smog_list)
    mean_smog = np.mean(smog_array)
    std_smog = np.std(smog_array)
    mean_data_cr = sum(data_cr_list)/len(data_cr_list)

    return (mean_data_cr, mean_smog, std_smog)


## Hyper Parameters of CSR Reward Function
mean_data_cr, mean_smog, std_smog = hyper_params_csr()
RHO = 0.3
LAMBDA = 2


def calc_similarity_reward(completions, reference, **kwargs):
    rewards = []

    for completion, ref in zip(completions, reference):

        # print(type(completion[0]['content']))
        # print()
        # print(type(ref))
        # exit()
        
        embeddings = sim_model.encode([completion[0]['content'], ref])

        score = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]
        score = max(0.0, min(1.0, score))
        rewards.append(score)

    return rewards

def calc_concise_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        prompt = prompt[0]['content']
        completion = completion[0]['content']
        report = prompt.split("### Report:")[-1].split("### Summary:")[0]

        # print(report)
        # print()
        # print(completion)
        
        len_report = len(llama_tok_concise_rew.encode(report))
        len_comp = len(llama_tok_concise_rew.encode(completion))

        # print(len_report, '   ', len_comp)

        # exit()
        
        gen_cr = len_report/len_comp
        diff = abs(gen_cr-mean_data_cr)
        reward = 1/(1+(diff/RHO)**LAMBDA)
        reward = max(0.0, min(1.0, reward))
        rewards.append(reward)

    return rewards

def calc_readability_reward(completions, **kwargs):

    # completions comes in chat template
    
    rewards = []
    for ii, completion in enumerate(completions):
        smog_id = textstat.smog_index(completion[0]['content'])
        rewards.append(np.exp(-abs(smog_id - mean_smog) / std_smog))

    return rewards

def reward_fn(prompts, completions, reference, **kwargs):

    # sim_rewards = calc_similarity_reward(completions, reference)
    # read_rewards = calc_readability_reward(completions)
    # con_rewards = calc_concise_reward(prompts,completions)
    
    # batch_rewards = [sim_rewards, read_rewards, con_rewards]

    return [random.random() for ii in range(len(prompts))]

    

if __name__ == '__main__':

    reward_fn(prompts=None, completions=None, reference=None)