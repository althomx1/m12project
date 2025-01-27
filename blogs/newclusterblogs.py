import nltk.cluster.util
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
cve_desc_file = pd.read_csv("../validation/cve_descriptions.csv")

blogs_file = pd.read_csv("../new web/results_blogs_no_preprocessing.csv")
# cve_desc_embedding = pd.DataFrame(cve_desc_file)
blogs_embeddings = pd.DataFrame(blogs_file)

blogs = blogs_embeddings['embedding']
# cves = cve_desc_embedding['embedding']

arr = []
#convert from string to array blogs
for emb_b in blogs:
    emb_b = emb_b.strip("[]")
    emb_list = [float(x) for x in emb_b.split()]
    emb_arr = np.array(emb_list)
    arr.append(emb_arr)


embs = np.array(arr)
# arr2 = []
# for emb_c in cves:
#     emb_c = emb_c.strip("[]")
#     emb_list_c = [float(x) for x in emb_c.split()]
#     emb_arr_c = np.array(emb_list_c)
#     arr2.append(emb_arr_c)
#
# cve_embs = np.array(arr2)

model = KMeansClusterer(2,distance=nltk.cluster.util.cosine_distance,avoid_empty_clusters=True,repeats=25)



assigned_clusters = model.cluster(embs, assign_clusters=True)

# assigned_clusters2 = model.cluster(cve_embs, assign_clusters=True)

# blogs_embeddings['cluster'] = pd.Series(assigned_clusters, index=blogs_embeddings.index)
#
# blogs_embeddings['centroid'] = blogs_embeddings['cluster'].apply(lambda x: model.means()[x])

# score = silhouette_score(embs, assigned_clusters, metric='cosine')
#
# print(score)
# print(assigned_clusters)
# for i in range(0, len(blogs_embeddings)):
#     val = distance_matrix([arr[i]], [blogs_embeddings['centroid'][i].tolist()])[0][0]
#     blogs_embeddings.loc[i, 'cosine_dist'] = val
#
#
# # blogs_embeddings.to_csv("results_blogs_with_cos_diff.csv", index = False)
# labels = blogs_embeddings['data'].tolist()
tsne_model = TSNE(perplexity=20, n_components=2, init='pca', max_iter=2500, random_state=42)
np.set_printoptions(suppress=True)
tsne = tsne_model.fit_transform(embs)
# tsne2 = tsne_model.fit_transform(cve_embs)

coord_x = tsne[:, 0]
coord_y = tsne[:, 1]

# coord_x_cve = tsne2[:, 0]
# coord_y_cve = tsne2[:, 1]

plt.figure(figsize=(35, 35))

plt.scatter(coord_x, coord_y, c=assigned_clusters, s=1500,alpha=.5)

# plt.scatter(coord_x_cve, coord_y_cve, c=assigned_clusters2, s=1500,alpha=.5)

sentences = blogs_embeddings['data']
# sentences_cve = cve_desc_embedding['data']
for j in range(len(sentences)):
    plt.annotate(
        f" ",
        xy=(coord_x[j], coord_y[j]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right', va='bottom')

# for k in range(len(sentences_cve)):
#     plt.annotate(
#         f"{str('*')}",
#         xy=(coord_x_cve[k], coord_y_cve[k]),
#         xytext=(5, 2),
#         textcoords='offset points',
#         ha='right', va='bottom')

plt.title("Clustering for raw text")
plt.show()




# model.fit(embs)
#
# labels_blogs = model.labels_
#
#
# text_with_labels = pd.DataFrame(columns=['text', 'label'])
# text_with_labels['text'] = blogs_embeddings['data']
# text_with_labels['label'] = labels_blogs
#
# cluster_0 = []
# cluster_1 = []
#
# for i, row in text_with_labels.iterrows():
#     if row['label'] == 0:
#         cluster_0.append(row['text'])
#     else:
#         cluster_1.append(row['text'])






# print(cluster_0)
# print()
# print()
# print(cluster_1)




























