import pandas as pd
import re
import transformers
from transformers import RobertaTokenizer,RobertaModel
import torch
from torch.nn.functional import normalize
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('punkt_tab')



file = pd.read_csv("../new web/blog posts6.csv")

result_df = pd.DataFrame(file)



punctuation = re.compile("(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")

result_df['cleaned_data'] = result_df['data'].apply(lambda x: punctuation.sub("",str(x)))
result_df['cleaned_data'] = result_df['cleaned_data'].str.rstrip('.')
def to_lower(X):
    X[X.columns[2]] = X[X.columns[2]].str.lower()
    return X
result_df = to_lower(result_df)

def clean_data(X):
    X[X.columns[2]] = X[X.columns[2]].str.replace(r"[^\w\s]|[\d]", "")
    return X

result_df = clean_data(result_df)

def tokenize(X):
    X[X.columns[2]] = X[X.columns[2]].apply(word_tokenize)
    return X

result_df = tokenize(result_df)

def remove_stopwords(X,stop):
    col_name = X.columns[2]
    X[col_name] = X[col_name].apply(lambda x: [w for w in x if w not in stop])

    return X

result_df = remove_stopwords(result_df, stop_words)

def stem(X):
    s = PorterStemmer()
    X[X.columns[2]] = X[X.columns[2]].apply(lambda x: [s.stem(word)for word in x])

    return X

result_df = stem(result_df)

def to_string(X):
    X[X.columns[2]] = X[X.columns[2]].apply(lambda x: ' '.join(words for words in x))
    return X

result_df = to_string(result_df)

# print(result_df['cleaned_data'].head())




# result_df['cleaned_data'] = result_df['cleaned_data'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# print(result_df['cleaned_data'])

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

for para in result_df['cleaned_data']:
    embedding = create_embedding(para, model, tokenizer)
    embeddings.append(embedding)
result_df['embedding'] = embeddings

result_df.to_csv("results_blogs6.csv", index=False)




