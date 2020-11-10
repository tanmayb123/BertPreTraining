import sys
from transformers import BertTokenizer, BertForMaskedLM
import torch

tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased" if sys.argv[1] == "orig" else "bert_lblq")

x = tok.encode(sys.argv[2])
idx = x.index(103)
x = torch.Tensor([x]).long()

l = model(x)[0]
l = l[0][idx].detach().cpu().numpy()
la = l.argsort()

for i in range(1, 11):
    print(tok.decode([la[-i]]))
