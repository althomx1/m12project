#libs
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import re
from transformers import RobertaTokenizer,RobertaModel
import torch
from torch.nn.functional import normalize
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

file = pd.read_csv("../validation/cve_description_unique.csv")
result = pd.DataFrame(file)

data = result['description']

#clean data
punctuation = re.compile("(\.)|(,)|(\;)|(\:)|(\!)|(\')|(\?)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\*)")
result['cleaned_data'] = result['description'].apply(lambda x: punctuation.sub("",x))
result['cleaned_data'] = result['cleaned_data'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


#create embeddings

tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = RobertaModel.from_pretrained("ehsanaghaei/SecureBERT")

def create_embedding(paragraph, model, tokenizer):

    inputs = tokenizer(paragraph,return_tensors="pt",truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:,0,:]

        normalized_embeddings = normalize(cls_embeddings, p=2,dim=1)

        embedding = normalized_embeddings.squeeze().cpu().numpy()

    return embedding


embeddings =[]

for para in result['cleaned_data']:
    embedding = create_embedding(para, model, tokenizer)
    embeddings.append(embedding)
result['embedding'] = embeddings


result.to_csv("cve_descriptions_emb_unique.csv", index=False)
