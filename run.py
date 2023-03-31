from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from utils import *
from rouge import Rouge

rouge = Rouge()

model_name = "../autodl-tmp/pertrained_model/chatglm-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

train_data = {"text": read_file("data/dataset/LCSTS/train.src.txt")[:10], "target": read_file("data/dataset/LCSTS/train.tgt.txt")[:10]}
test_data = {"text": read_file("data/dataset/LCSTS/test.src.txt")[:10], "target": read_file("data/dataset/LCSTS/test.tgt.txt")[:10]}
begin = "我给你一些文本摘要的示例模版供你学习，然后根据学习的内容进行文本摘要生成"
response, history = model.chat(tokenizer, begin, history=[])
prompt_train = "文本："
prompt_train_label = "摘要："
for t, l in zip(train_data["text"], train_data["target"]):
    response, history = model.chat(tokenizer, prompt_train+t+' '+prompt_train_label+l, history=[])
    print(response)
print("generactor text summary.")
# prompt = "请给下面一段文本做摘要："
predict = []
for t in test_data['text']:
    response, history = model.chat(tokenizer, prompt_train+t+' '+prompt_train_label, history=[])
    predict.append(' '.join(list(response.strip())))
    print(response)
target = [' '.join(list(tar)) for tar in test_data['target']]
avg_rouge_scores = rouge.get_scores(predict, target, avg=True)
print(avg_rouge_scores)
