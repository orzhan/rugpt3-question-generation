import numpy as np
import random
import json
train_data = json.load(open("train-v1.1.json"))
valid_data = json.load(open("dev-v1.1.json"))
train = []
for item in train_data['data'][0]['paragraphs'] + valid_data['data'][0]['paragraphs']:
  train.append('Context: ' + item['context'] + ' Answer: '+ item['qas'][0]['answers'][0]['text'] + ' Question: ' + item['qas'][0]['question']  )

train2 = []
for item in train_data['data'][0]['paragraphs'] + valid_data['data'][0]['paragraphs']:
  train2.append('Context: ' + item['context'] + ' Question: ' + item['qas'][0]['question'] + ' Answer: '+ item['qas'][0]['answers'][0]['text'] )

with open("train2.txt", "w") as file:
    file.write("\n".join(train2))

with open("train.txt", "w") as file:
    file.write("\n".join(train))
