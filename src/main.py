import torch
from KAARMA import *
from tomita import generate_tomita, generate_tomita_sequence
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


tomita_type = 4
model = KAARMA(4, 1, 2, 2)
model.node.load_state_dict(torch.load('model/n_%d.pkl' % tomita_type))
model.eval()
np.random.seed(2)
strings, desires = generate_tomita_sequence(100, 2000, tomita_type)

res, state_trajectory = model(strings, True)
state_trajectory = state_trajectory.detach().view(-1, 4).data.numpy()
res = res > 0.5
print(np.sum(res.data.numpy() == desires))
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(state_trajectory)

kmeans = KMeans(n_clusters=8).fit(state_trajectory)
X_centers = pca.transform(kmeans.cluster_centers_)
# np.save('trajectory/%d.npy' % tomita_type, kmeans.cluster_centers_)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.scatter(X_centers[:, 0], X_centers[:, 1])
plt.show()