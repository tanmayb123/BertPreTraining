import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertForMaskedLM

class PTDataset(Dataset):
    def __init__(self):
        self.data = pickle.load(open("lblq_data.pkl", "rb"))
        idxs = list(range(len(self.data[0])))
        random.shuffle(idxs)
        self.data = [[x[i] for i in idxs] for x in self.data]
        print("Loaded {} rows of data".format(len(self.data[0])))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [torch.tensor(x[idx]) for x in self.data]

class BertLyricsPT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def forward(self, x):
        inp, out, mask = x
        enc = self.bert(input_ids=inp, attention_mask=mask, labels=out)
        return enc

    def training_step(self, batch, batch_nb):
        yhat = self(batch)
        loss = yhat.loss
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5, eps=1e-4)

if __name__ == "__main__":
    trainer = pl.Trainer(gpus=4, distributed_backend='ddp', max_epochs=4)

    dataset = PTDataset()
    dataloader = DataLoader(dataset, batch_size=None)

    print("Loading model")
    model = BertLyricsPT()
    print("Loaded model")

    trainer.fit(model, dataloader)
    model.bert.save_pretrained("bert_lblq")
