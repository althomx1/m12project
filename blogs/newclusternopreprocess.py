import nltk.cluster.util
from nltk.cluster import KMeansClusterer
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

blogs_file = pd.read_csv("../new web/results_blogs2.csv")

blogs_embeddings = pd.DataFrame(blogs_file)

blogs = blogs_embeddings['embedding']

arr = []
#convert from string to array blogs
for emb_b in blogs:
    emb_b = emb_b.strip("[]")
    emb_list = [float(x) for x in emb_b.split()]
    emb_arr = np.array(emb_list)
    arr.append(emb_arr)


embs = np.array(arr)


model = KMeansClusterer(2,distance=nltk.cluster.util.cosine_distance,avoid_empty_clusters=True,repeats=25)



assigned_clusters = model.cluster(embs, assign_clusters=True)

blogs_embeddings['cluster'] = pd.Series(assigned_clusters, index=blogs_embeddings.index)

blogs_embeddings['centroid'] = blogs_embeddings['cluster'].apply(lambda x: model.means()[x])

# score = silhouette_score(embs, assigned_clusters, metric='cosine')
#
# print(score)
# print(assigned_clusters)
# for i in range(0, len(blogs_embeddings)):
#     val = distance_matrix([arr[i]], [blogs_embeddings['centroid'][i].tolist()])[0][0]
#     blogs_embeddings.loc[i, 'cosine_dist'] = val


# blogs_embeddings.to_csv("results_blogs_with_cos_diff.csv", index = False)
# labels = blogs_embeddings['data'].tolist()
# model = TSNE(perplexity=20, n_components=2, init='pca', max_iter=2500, random_state=23)
# np.set_printoptions(suppress=True)
# tsne = model.fit_transform(embs)
# coord_x = tsne[:, 0]
# coord_y = tsne[:, 1]
#
# plt.figure(figsize=(35, 35))
# plt.scatter(coord_x, coord_y, c=assigned_clusters, s=1500,alpha=.5)
# sentences = blogs_embeddings['data']
# for j in range(len(sentences)):
#     plt.annotate(
#         f"{str(assigned_clusters[j])}" ,
#         xy=(coord_x[j], coord_y[j]),
#         xytext=(5, 2),
#         textcoords='offset points',
#         ha='right', va='bottom')
#
# plt.show()






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




























