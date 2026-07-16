from evaluate import load as eval_load
from datasets import load_dataset
from tqdm import tqdm

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

    with open(file_name,'w') as f:
        f.write(f"ROUGE-L: {rouge_res['rougeL']}\n")
        f.write(f"BERTScore-F1: {sum(bert_res['f1']) / len(bert_res['f1'])}")

if __name__ == '__main__':
    preds, refs = [], []
    test_dataset = load_dataset('csv',data_files='test.csv')['train']
    pred_dir = 'output_awgrpo/awgrpo_outputs_200'
    for ii,sample in enumerate(tqdm(test_dataset,total=len(test_dataset),desc='Converting samples for eval')):
        # print(sample)
        # break
        pred, ref = '', sample['GPT_Summary']
        with open(f'{pred_dir}/{ii}.txt', 'r') as f:
            pred = f.read()
        # print()
        # print(pred)
        # print()
        # print(ref)
        # break
        preds.append(pred)
        refs.append(ref)
        
    eval_score(preds,refs,f'{pred_dir.split('/')[0]}/{pred_dir.split('/')[1]}_eval_score.txt')