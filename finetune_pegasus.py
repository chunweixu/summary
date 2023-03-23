import dict
from utils import *
from tokenizers_pegasus import PegasusTokenizer
from transformers import PegasusForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 加载数据集
dataset = {'doc': read_file(dict.finetune_pegasus_train_text)[:10000], 'summary': read_file(dict.finetune_pegasus_train_label)[:10000]}

# 加载预训练模型和分词器
model_name = dict.pegasus_model_path
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# 使用 map() 方法进行数据处理
@calculate_time
def preprocess_function(examples):
    res = []
    for inputs, targets in zip(examples['doc'], examples['summary']):
        inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
        res.append({"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "decoder_input_ids": targets["input_ids"], "decoder_attention_mask": targets["attention_mask"], "labels": targets["input_ids"]})
    return res

dataset = preprocess_function(dataset)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 定义微调参数
training_args = Seq2SeqTrainingArguments(
    output_dir=dict.train_model_template,
    evaluation_strategy = "steps",
    eval_steps = 100,
    save_steps = 500,
    num_train_epochs = 1,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    logging_steps = 100,
    learning_rate = 1e-4,
    warmup_steps = 500,
    predict_with_generate=True,
)

# 定义微调器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 进行微调
trainer.train()

model.save_pretrained(dict.save_finetune_pegasus_summary)