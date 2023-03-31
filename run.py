from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from utils import *
from rouge import Rouge

rouge = Rouge()

model_name = "../autodl-tmp/pertrained_model/chatglm-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

train_data = {"text": read_file("data/dataset/LCSTS/train.src.txt")[:20], "target": read_file("data/dataset/LCSTS/train.tgt.txt")[:20]}
test_data = {"text": read_file("data/dataset/LCSTS/test.src.txt")[:1000], "target": read_file("data/dataset/LCSTS/test.tgt.txt")[:1000]}
# begin = "请生成文本的标题"
# response, history = model.chat(tokenizer, begin, history=[])
# prompt_train = "文本："
# prompt_train_label = "标题："
# for t, l in zip(train_data["text"], train_data["target"]):
#     # response, history = model.chat(tokenizer, prompt_train+t+' '+prompt_train_label+l, history=[])
#     response, history = model.chat(tokenizer, t + " => " + l, history=[])
#     print(response)
print("generactor text title.")
prompt = "请直接给出下面文本的标题："
predict = []
for t in test_data['text']:
    # response, history = model.chat(tokenizer, prompt_train+t+' '+prompt_train_label, history=[])
    response, history = model.chat(tokenizer, prompt+t, history=[])
    # response, history = model.chat(tokenizer, prompt + " => ", history=[])
    predict.append(' '.join(list(response.strip())))
    print(response)
target = [' '.join(list(tar)) for tar in test_data['target']]
avg_rouge_scores = rouge.get_scores(predict, target, avg=True)
print(avg_rouge_scores)
