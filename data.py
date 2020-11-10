import math
import pickle
import random
import multiprocessing as mp
from transformers import BertTokenizer
from tqdm import tqdm

flatten = lambda t: [item for sublist in t for item in sublist]

tok = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 15
BATCH_SIZE = 128

def tokenize(x):
    return tok.encode(x)[1:-1]

def process(_n):
    n = _n
    size = math.ceil(len(n)*0.15)
    indices = random.sample(range(len(n)), size)
    replacement_prob = [True, True, True, True, True, True, True, True, False, False]
    output = [-100]*len(n)
    for index in indices:
        output[index] = n[index]
        if random.choice(replacement_prob):
            n[index] = 103
        else:
            if random.choice([True, False]):
                n[index] = random.randint(1010, 30500)
    return (n, output)

def batcher(ins, outs):
    batchx, batchy, batchesx, batchesy, batchesmasks = [], [], [], [], []
    for i in zip(ins, outs):
        batchx.append(i[0])
        batchy.append(i[1])
        if len(batchx) == BATCH_SIZE:
            maxlen = max([len(x) for x in batchx])
            masks = [[1] * len(x) + [0] * (maxlen - len(x)) for x in batchx]
            batchx = [x + [0] * (maxlen - len(x)) for x in batchx]
            batchy = [x + [0] * (maxlen - len(x)) for x in batchy]
            batchesx.append(batchx)
            batchesy.append(batchy)
            batchesmasks.append(masks)
            batchx = []
            batchy = []
    return [batchesx, batchesy, batchesmasks]

# real_data is an array of strings containing multiple lines, each line is used as its own sample later in the script
real_data = list(set(flatten([x.split("\n") for x in open("../full_verses.txt").read().split("\n\n")])))
print(len(real_data))
random.shuffle(real_data)

pool = mp.Pool()
tokenized = list(tqdm(pool.imap(tokenize, real_data), total=len(real_data)))

inputs = tokenized
outputs = list(tqdm(pool.imap(process, tokenized), total=len(tokenized)))
outputs.sort(key=lambda x: len(x[0]))

inputs = [[101] + x[0] + [102] for x in outputs]
outputs = [[-100] + x[1] + [-100] for x in outputs]

batched = batcher(inputs, outputs)
pickle.dump(batched, open("lblq_data.pkl", "wb"))
