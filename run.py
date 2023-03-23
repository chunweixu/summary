from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import torch

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the test dataset
test_dataset = [...]  # Fill in the test dataset

# Define a function to calculate ROUGE metrics
def evaluate_rouge(model, tokenizer, dataset):
    model.eval()

    rouge = Rouge()

    with torch.no_grad():
        for example in dataset:
            # Get the input and target summary
            input_text = example['input_text']
            target_summary = example['target_summary']

            # Encode the input and target summary using the tokenizer
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            target_ids = tokenizer.encode(target_summary, return_tensors='pt')

            # Generate a summary
            summary_ids = model.generate(input_ids)

            # Decode the generated summary into text
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_summary = generated_summary.strip()

            # Calculate the ROUGE metrics
            scores = rouge.get_scores(generated_summary, target_summary)
            rouge_scores['rouge-1']['f'].append(scores[0]['rouge-1']['f'])
            rouge_scores['rouge-1']['p'].append(scores[0]['rouge-1']['p'])
            rouge_scores['rouge-1']['r'].append(scores[0]['rouge-1']['r'])
            rouge_scores['rouge-2']['f'].append(scores[0]['rouge-2']['f'])
            rouge_scores['rouge-2']['p'].append(scores[0]['rouge-2']['p'])
            rouge_scores['rouge-2']['r'].append(scores[0]['rouge-2']['r'])
            rouge_scores['rouge-l']['f'].append(scores[0]['rouge-l']['f'])
            rouge_scores['rouge-l']['p'].append(scores[0]['rouge-l']['p'])
            rouge_scores['rouge-l']['r'].append(scores[0]['rouge-l']['r'])

    # Calculate the average ROUGE metrics
    avg_rouge_scores = {}
    for metric, results in rouge_scores.items():
        avg_rouge_scores[metric] = {'f': sum(results['f']) / len(results['f']),
                                    'p': sum(results['p']) / len(results['p']),
                                    'r': sum(results['r']) / len(results['r'])}

    return avg_rouge_scores

# Calculate ROUGE metrics on the test set and output the results
rouge_scores = evaluate_rouge(model, tokenizer, test_dataset)
print(pd.DataFrame.from_dict(rouge_scores)))



# 加载Pegasus模型和tokenizer
model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# 加载测试数据集
test_dataset = [...]  # 填写测试数据集

# 定义函数来计算ROUGE指标
def evaluate_rouge(model, tokenizer, dataset):
    model.eval()
    rouge = Rouge()

    with torch.no_grad():
        for example in dataset:
            # 获取输入和目标摘要
            input_text = example['input_text']
            target_summary = example['target_summary']

            # 使用tokenizer编码输入和目标摘要
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            target_ids = tokenizer.encode(target_summary, return_tensors='pt')

            # 生成摘要
            summary_ids = model.generate(input_ids)

            # 将生成的摘要解码为文本
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generated_summary = generated_summary.strip()

            # 计算ROUGE指标
            scores = rouge.get_scores(generated_summary, target_summary)
            rouge_scores['rouge-1']['f'].append(scores[0]['rouge-1']['f'])
            rouge_scores['rouge-1']['p'].append(scores[0]['rouge-1']['p'])
            rouge_scores['rouge-1']['r'].append(scores[0]['rouge-1']['r'])
            rouge_scores['rouge-2']['f'].append(scores[0]['rouge-2']['f'])
            rouge_scores['rouge-2']['p'].append(scores[0]['rouge-2']['p'])
            rouge_scores['rouge-2']['r'].append(scores[0]['rouge-2']['r'])
            rouge_scores['rouge-l']['f'].append(scores[0]['rouge-l']['f'])
            rouge_scores['rouge-l']['p'].append(scores[0]['rouge-l']['p'])
            rouge_scores['rouge-l']['r'].append(scores[0]['rouge-l']['r'])

    # 计算平均ROUGE指标
    avg_rouge_scores = {}
    for metric, results in rouge_scores.items():
        avg_rouge_scores[metric] = {'f': sum(results['f']) / len(results['f']),
                                    'p': sum(results['p']) / len(results['p']),
                                    'r': sum(results['r']) / len(results['r'])}

    return avg_rouge_scores