
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


file_desc = pd.read_csv("../validation/cve_descriptions_emb_unique.csv")
embedd = file_desc['embedding']


arr = []
for emb_b in embedd:
    emb_b = emb_b.strip("[]")  # Remove brackets
    emb_list = [float(x) for x in emb_b.split()]  # Convert to list of floats
    arr.append(np.array(emb_list))  # Append numpy array

# Compute centroid
embs = np.array(arr)
centroid = np.mean(embs, axis=0)


# Load blog embeddings
blogs_file = pd.read_csv("../new web/results_blogs_no_preprocessing.csv")
blogs = blogs_file['embedding']

# Calculate cosine distance
distances = []
for emb_blog in blogs:
    emb_blog = emb_blog.strip("[]")  # Remove brackets
    emb_list = [float(x) for x in emb_blog.split()]  # Convert to list of floats
    emb_arr = np.array(emb_list)  # Convert to numpy array
    distance = cosine(emb_arr, centroid)  # Compute cosine distance
    distances.append(distance)

# Add distances to a new DataFrame
results_distance = pd.DataFrame({
    'distance': distances
})


df_combined = pd.concat([blogs_file['data'],results_distance],axis=1)


df_combined.to_csv("results_distance_with_text_2.csv",index=False)