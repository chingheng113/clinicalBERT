import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch import nn
import math
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from downstream_tasks.run_classifier import BertForMultiLabelSequenceClassification
current_path = os.path.dirname(__file__)
import seaborn as sns
sns.set_style('white')
sns.set_context('talk', font_scale = 1)

def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


def get_attention_scores(model, i, text):
    tokenized = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)

    segment_ids = [0] * len(indexed_tokens)
    t_tensor = torch.tensor([indexed_tokens])
    s_ids = torch.tensor([segment_ids])

    outputs_query = []
    outputs_key = []

    def hook_query(module, input, output):
        # print ('in query')
        outputs_query.append(output)

    def hook_key(module, input, output):
        # print ('in key')
        outputs_key.append(output)

    model.bert.encoder.layer[i].attention.self.query.register_forward_hook(hook_query)
    model.bert.encoder.layer[i].attention.self.key.register_forward_hook(hook_key)
    l = model(t_tensor, s_ids)

    query_layer = transpose_for_scores(bert_config, outputs_query[0])
    key_layer = transpose_for_scores(bert_config, outputs_key[0])

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(int(bert_config.hidden_size / bert_config.num_attention_heads))
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    return attention_probs, tokenized

# ===============================
# bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'carotid', 'sb_all_0', 'config.json'))
# tokenizer = BertTokenizer.from_pretrained(os.path.join(current_path, 'models', 'strokeBERT_biobased_all_150000'), do_lower_case=False)

bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'restroke_all', 'sb_all_0', 'config.json'))
tokenizer = BertTokenizer.from_pretrained(os.path.join(current_path, 'models', 'biobert_v1.0_pubmed_pmc'), do_lower_case=False)

model = BertForMultiLabelSequenceClassification(bert_config, num_labels=17)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text ='significant stenosis at right MCA M1 segment and right distal VA'
text = "This is a male with past history of old CVA with Lt weakness 5 years ago. HTN with OPD medication control for 5 years. He became totally recovered without neurologic deficit after the old CVA."
x, tokens = get_attention_scores(model, 0, text)

map1=np.asarray(x[0][1].detach().numpy())


plt.clf()

f=plt.figure(figsize=(10,10))
ax = f.add_subplot(1,1,1)
i=ax.imshow(map1, interpolation='nearest', cmap='Purples')
# f.colorbar(i, ax=ax)

ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens)

ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens,rotation=90)

ax.set_xlabel('key')
ax.set_ylabel('query')

ax.grid(linewidth = 0.8)

plt.show()