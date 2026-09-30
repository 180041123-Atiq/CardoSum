from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import numpy as np
import textstat
import pandas as pd
import random
from datasets import load_dataset
from tqdm import tqdm

# will try to use the tokenizer the model is using for tokenization
llama_tok_concise_rew = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
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
        # print(completion[0]['content'])
        # print()
        # print(type(ref))
        # print(ref)
        # exit()
        if isinstance(completion,str):
            embeddings = sim_model.encode([completion, ref])
        else:
            embeddings = sim_model.encode([completion[0]['content'], ref])

        score = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]
        score = max(0.0, min(1.0, score))
        rewards.append(score)

    return rewards

def calc_concise_reward(prompts, completions, **kwargs):
    rewards = []
    initial_prompt = 'Please explain the following Echocardiography report. '
    for prompt, completion in zip(prompts, completions):
        if not isinstance(prompt,str):
            prompt = prompt[0]['content']
        if not isinstance(completion, str):
            completion = completion[0]['content']
        report = prompt.split(initial_prompt)[1]

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



def calc_piecewise_conciseness_reward(prompts, completions, **kwargs):

    rewards = []
    initial_prompt = 'Please explain the following Echocardiography report. '

    target_cr = 1.6

    # Reward at CR = 1.0
    base_reward = 0.70

    # Maximum reward at target CR
    max_reward = 1.0

    # Controls how gradually reward rises from 1.0 -> 1.6
    exponent = 2.5

    # Penalty after target
    upper_scale = 0.20

    for prompt, completion in zip(prompts, completions):

        if not isinstance(prompt, str):
            prompt = prompt[0]['content']

        if not isinstance(completion, str):
            completion = completion[0]['content']

        report = prompt.split(initial_prompt)[1]

        len_report = len(
            llama_tok_concise_rew.encode(report)
        )

        len_comp = len(
            llama_tok_concise_rew.encode(completion)
        )

        if len_comp == 0:
            rewards.append(0.0)
            continue

        gen_cr = len_report / len_comp

        if gen_cr < 1.0:

            # Penalize excessive verbosity
            # but keep the penalty relatively mild.
            reward = base_reward * (gen_cr ** 2)

        elif gen_cr <= target_cr:

            # Normalize CR from [1.0, 1.6] -> [0, 1]
            progress = (gen_cr - 1.0) / (target_cr - 1.0)

            # Diminishing returns as CR approaches 1.6
            reward = (
                base_reward
                + (max_reward - base_reward)
                * (progress ** exponent)
            )

        else:

            # Strong penalty for excessive compression
            diff = gen_cr - target_cr

            reward = max_reward * np.exp(
                -diff / upper_scale
            )

        reward = np.clip(reward, 0.0, 1.0)

        rewards.append(float(reward))

    return rewards



def calc_readability_reward(completions, **kwargs):

    # completions comes in chat template
    
    rewards = []
    for ii, completion in enumerate(completions):
        # print(type(completion))
        # exit()
        if isinstance(completion,str):
            smog_id = textstat.smog_index(completion)
        else:
            smog_id = textstat.smog_index(completion[0]['content'])
        rewards.append(np.exp(-abs(smog_id - mean_smog) / std_smog))

    return rewards


def calc_piecewise_readability_reward(completions, **kwargs):

    # Desired readability target
    target_smog = 6.0

    # Penalty for readability scores below the target
    lower_scale = 2.0

    # Very small penalty between target and mean
    near_target_scale = 10.0

    # Stronger penalty beyond the mean
    upper_scale = 1.0

    rewards = []

    for ii, completion in enumerate(completions):

        if isinstance(completion, str):
            text = completion
        else:
            text = completion[0]['content']

        smog_id = textstat.smog_index(text)

        if smog_id <= target_smog:
            # Penalize scores below the target
            reward = np.exp(
                -(target_smog - smog_id) / lower_scale
            )

        elif smog_id <= mean_smog:
            # Very small penalty between target and mean
            reward = np.exp(
                -(smog_id - target_smog) / near_target_scale
            )

        else:
            # Stronger penalty once readability becomes
            # substantially worse than the population mean
            reward_at_mean = np.exp(
                -(mean_smog - target_smog) / near_target_scale
            )

            reward = reward_at_mean * np.exp(
                -(smog_id - mean_smog) / upper_scale
            )

        rewards.append(reward)

    return rewards


def reward_fn(prompts, completions, reference, **kwargs):

    # sim_rewards = calc_similarity_reward(completions, reference)
    # read_rewards = calc_readability_reward(completions)
    # con_rewards = calc_concise_reward(prompts,completions)
    
    # batch_rewards = [sim_rewards, read_rewards, con_rewards]

    return [random.random() for ii in range(len(prompts))]

    

if __name__ == '__main__':
    
    prompts,completions,references = [],[],[]
    test_dataset = load_dataset("csv",data_files="test.csv",)['train']
    
    # output_dir = 'output_awgrpo/grpo_outputs'
    # output_dir = 'output_awgrpo/awgrpo_outputs'
    # output_dir = 'output_awgrpo/sft_outputs'
    # output_dir = 'output_awgrpo/icl2_outputs/icl2_outputs'
    # output_dir = 'output_awgrpo/icl2_outputs/meditron7b_icl2_outputs'
    # output_dir = 'output_awgrpo/icl2_outputs/vicuna7b_icl2_outputs'
    # output_dir = 'output_awgrpo/llama8b_icl2_outputs'
    output_dir = 'gen_outputs_HVO'
    
    for ii,sample in enumerate(tqdm(test_dataset, total=len(test_dataset), desc='Converting samples for reward calc')):
        prompts.append(sample['Original_Report'])
        references.append(sample['GPT_Summary'])
        completion = ''
        with open(f'{output_dir}/{ii}.txt','r') as f:
            completion = f.read()
        
        completions.append(completion)

    # For Normal Rewards
    # readability_rewards = calc_readability_reward(completions)
    # conciseness_rewards = calc_concise_reward(prompts,completions)
    # similarity_rewards = calc_similarity_reward(completions,references)

    # For Piecewise Readability and Conciseness
    readability_rewards = calc_piecewise_readability_reward(completions)
    conciseness_rewards = calc_piecewise_conciseness_reward(prompts,completions)
    similarity_rewards = calc_similarity_reward(completions,references)
    
    print('avg readability', sum(readability_rewards)/len(readability_rewards))
    print('avg conciseness', sum(conciseness_rewards)/len(conciseness_rewards))
    print('avg similarity', sum(similarity_rewards)/len(similarity_rewards))
    # reward_fn(prompts=None, completions=None, reference=None)

    with open(f'results/{output_dir}_reward_score.txt','w') as f:
        f.write(f"avg readability: {sum(readability_rewards)/len(readability_rewards)}\n")
        f.write(f"avg conciseness: {sum(conciseness_rewards)/len(conciseness_rewards)}\n")
        f.write(f"avg similarity: {sum(similarity_rewards)/len(similarity_rewards)}")