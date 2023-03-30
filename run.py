from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from utils import *
from rouge import Rouge

rouge = Rouge()

model_name = "../autodl-tmp/pertrained_model/chatglm-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

test_data = {"text": read_file("data/dataset/LCSTS/test.src.txt")[:1000], "target": read_file("data/dataset/LCSTS/test.tgt.txt")[:1000]}
prompt = "请给下面一段文本做摘要："
predict = []
for t in test_data['text']:
    response, history = model.chat(tokenizer, prompt+t, history=[])
    predict.append(' '.join(list(response.strip())))
    print(response)
target = [' '.join(list(tar) for tar in test_data['target'])]
avg_rouge_scores = rouge.get_scores(predict, target, avg=True)
print(avg_rouge_scores)
