import torch
from KAARMA import *
from tomita import generate_tomita, generate_tomita_sequence
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


tomita_type = 7
model = KAARMA(4, 1, 2, 2)
model.node.load_state_dict(torch.load('../model/%d.pkl' % tomita_type))
model.eval()

strings, desires = generate_tomita_sequence(1, 200, tomita_type)

res, state_trajectory = model(strings[0], True)
res = res > 0.5
print(res == desires)
X_reduced = PCA(n_components=2).fit_transform(state_trajectory)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()
kmeans = KMeans(n_clusters=7).fit(state_trajectory)

np.save('../trajectory/%d.npy' % tomita_type, kmeans.cluster_centers_)
