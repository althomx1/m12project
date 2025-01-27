import pandas as pd
import re
import transformers
from transformers import RobertaTokenizer,RobertaModel
import torch
from torch.nn.functional import normalize
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))


file = pd.read_csv("../new web/blog posts6.csv")

result_df = pd.DataFrame(file)

# ab_pattern = r'(?<![A-Z][ar-z])(?<!\d)([^\w\s])|([^\w\s])(?![A-Z]|[a-z]{2}|\d)'
# ab_pattern = re.compile(ab_pattern)
#
# punctuation = re.compile("(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")

# result_df['cleaned_data'] = result_df['data'].apply(lambda x: punctuation.sub("",str(x)))
#
# result_df['cleaned_data'] = result_df['cleaned_data'].str.rstrip('.')
# result_df['cleaned_data'] = result_df['cleaned_data'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#
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

for para in result_df['data']:
    embedding = create_embedding(str(para), model, tokenizer)
    embeddings.append(embedding)
result_df['embedding'] = embeddings

result_df.to_csv("results_blogs_no_preprocessing.csv", index=False)



