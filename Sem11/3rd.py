import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


data = pd.read_csv("maisAssistidos.csv")
X = data.iloc[:,1:]
Y = data['Nome do filme']
X = X.replace(to_replace =['?'],value=-1)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
print("Agrupamento Herárquico",model.labels_)

plt.title('Dendrograma')
plot_dendrogram(model, truncate_mode='level')
plt.xlabel("Número do Filme")
plt.show()