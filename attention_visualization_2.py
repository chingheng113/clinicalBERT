import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch import nn
import math
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from downstream_tasks.run_classifier import BertForMultiLabelSequenceClassification, BertForSequenceClassification
current_path = os.path.dirname(__file__)
import seaborn as sns
sns.set_style('white')
sns.set_context('talk', font_scale = 1)
import matplotlib.cm as cm
import matplotlib.colors as colors

# https://github.com/shreydesai/attention-viz

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
tokenizer = BertTokenizer.from_pretrained(os.path.join(current_path, 'models', 'strokeBERT_biobased_all_150000'), do_lower_case=False)
# tokenizer = BertTokenizer.from_pretrained(os.path.join(current_path, 'models', 'biobert_pretrain_output_all_notes_150000'), do_lower_case=False)

# carotid
bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'carotid', 'sb_all_0', 'config.json'))
# bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'carotid', 'c_all_0', 'config.json'))
model = BertForMultiLabelSequenceClassification(bert_config, num_labels=17)
text = ' Subacute ischemic infarction, presented as high signal intensity on T2WI, FLAIR images and DWI images, with gyral enhancement is noted in right fronto-temporal lobes and right insular lobe. One tiny old brain insult with old hemosiderin deposition is noted over left periventricular region. No definite abnormal signal intensity mass lesion in the brain noted, including supratentorial cerebral hemisphere, infratentorial cerebellum and brain stem region. Symmetrical appearance of bil. lateral ventricles without apparent dilatation. No evidence of abnormal enhancing mass lesion, or abnormal leptomeningeal enhancement in the brain. Noncontrast intracranial MRA with 3D TOF shows that high grade stenosis over intracranial portion of vertebral basilar artery, right middle cerebral artery and bilateral posterior cerebral arteries (PCA). Right A1 segment is invisible and single anterior cerebral artery(ACA) is found. R/O occlusion of right anterior cerebral artery (ACA) or azygos anterior cerebral artery.'
# text ='significant stenosis at right MCA M1 segment and right distal VA'

# recurrent stroke
# bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'restroke_all', 'sb_all_0', 'config.json'))
# bert_config  = BertConfig(os.path.join(current_path, 'downstream_tasks', 'output', 'carotid', 'c_all_0', 'config.json'))
# model = BertForSequenceClassification(bert_config, num_labels=2)
# text = "Acute onset of L't side weakness with stationary course for 3 days. This 75-year-old female with history of 1.Congestive heart failure, New York Heart Association functional class IV II, ischemic cardiomyopathy with pulmonary edema. 2.Single vessel coronary artery disease[left anterior descending artery] status post successful percutaneous coronary intervention with drug-eluting-stent at left anterior descending artery. 3.Hypertension . 4.Diabetes mellitus. 5.Chronic kidney disease, stage 3. 6.Hypokalemia, favor diuretic related was admitted via ED due to Acute onset of L't side weakness with stationary course for 3 days.According to the patient, her ADL is partially dependent[walk with cane, can deal with her daily life]. This time, sudden onset of L't weakness was noticed on 11/10 while she was cooking lunch. The associate symptom include slurred speech. Through the course, there was no loss of consciousness, palpitation, chest pain, short of brath, cold sweating, diplopia, dysphagia, facial drooling, aphasia, agnosia nor parethesia. She visited our ED on 11/12 where initial vital signs were relative stable and no ICH was noticed on CT scan. She was admitted for further survey and treatment. After admission, brain MRI indicated a mild pons infarction.Left limbs weakness improved a lot. Due to chronic kidney disease, stage 3, adequate hydration with metformin discontinued was advised.She might be at risk of fall with not yet adequate goal of rehabilitation program.Due to an improved course, she asked for discharged home with independent life style with surveillance advised.Scheduled visit to Department of Neurology and Cardiology were scheduled."


x, tokens = get_attention_scores(model, 0, text)
scores = np.asarray(x[0][1].detach().numpy())[0]
cmap = cm.get_cmap('BuGn')

off = (sum(scores) / len(scores)) * 0.0
normer = colors.Normalize(vmin=min(scores)-off, vmax=max(scores)+off)
colors = [colors.to_hex(cmap(normer(x))) for x in scores]

if len(tokens) != len(colors):
    raise ValueError("number of tokens and colors don't match")

style_elems = []
span_elems= []
for i in range(len(tokens)):
    style_elems.append(f'.c{i} {{ background-color: {colors[i]}; }}')
    span_elems.append(f'<span class="c{i}">{tokens[i]} </span>')

print(
    f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>span {{ font-family: "Roboto Mono", monospace; font-size: 12px; }} {' '.join(style_elems)}</style></head><body>{' '.join(span_elems)}</body></html>"""
)
