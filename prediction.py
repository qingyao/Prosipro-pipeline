import torch, os, gc
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pandas as pd
import argparse
from datetime import date

today = date.today().strftime("%Y%m%d")
parser = argparse.ArgumentParser(
                    prog='Project selection in Proteomics',
                    description='It runs an fined-tuned LLM classifier against the title, abstract and keywords and outputs binary decision to include the project or not')
parser.add_argument('-m', '--modelpath', default = './best_model')
parser.add_argument('-t', '--textpath', default=f'./input/{today}.tsv', help='tsv with 6 columns: PX-ID, PMID, publication title, dataset title, keywords and abstract')
parser.add_argument('-n', '--modeltype', choices=['WordPiece','BPE'],default='BPE')

args = parser.parse_args()
# for prediction on Github Action only CPU
# ~100 texts/min
# tested with cuda 10x faster
device = "cpu" 
DEVICE = torch.device(device)
model_dir = args.modelpath # 
text_path = args.textpath
model_type = args.modeltype
metrics = pd.read_csv(os.path.join(model_dir, 'metrics.tsv'), sep='\t')
threshold = metrics['threshold'][0]
max_length = metrics['max_length'][0]
model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

text_list = []
id_list = []
with open(f'output/{today}.log', 'w') as logf:
    with open(text_path) as f:
        for i, l in enumerate(f):
            if model_type == 'WordPiece':
                sep = '[SEP]'
                start = '[CLS]'
            else:
                sep = '</s>'
                start = '<s>'
            
            ll = l.strip().split('\t')
            pxdid, pmid = ll[:2]
            try:
                text = start + ' Title: ' + ll[2] + \
                        sep + ' Dataset Title: ' + ll[3] + \
                        sep + ' Keywords: ' + ll[4] + \
                        sep + ' Abstract: ' + ll[5] + \
                        sep        
                # text = start + sep.join(l.strip().split('\t')) + sep
                text_list.append(text)
                id_list.append(pxdid+'\t'+pmid)
            except IndexError:
                print(i, l, sep = '\t', file = logf)

all_preds = []
all_ids = []
for i in range(len(text_list)):
    enc = tokenizer(text_list[i],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)

    logits = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    return_dict=True
                    ).logits
    probs = torch.softmax(torch.tensor(logits.to('cpu')), dim=1)[:,1].numpy()
    all_preds += (probs > threshold).tolist()
    all_ids += id_list[i]
    del enc
    gc.collect()

with open(f'output/{today}.txt', 'w') as f:
    for i, pred in enumerate(all_preds):
        print(all_ids[i], pred, sep = '\t', file = f)