import numpy as np

import torch
from mpl_toolkits.mplot3d import Axes3D
from KAARMA import *
from tomita import generate_tomita, generate_tomita_sequence
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


tomita_type = 6
model = KAARMA(4, 1, 2, 2)
model.node.load_state_dict(torch.load('../model/nn_%d.pkl' % tomita_type))
model.eval()
# np.random.seed(2)
strings, desires = generate_tomita_sequence(2000, 100, tomita_type)

strings = torch.from_numpy(strings.astype(np.float32))
desires = torch.from_numpy(desires.astype(np.float32))

res, state_trajectory = model(strings, True)
state_trajectory = state_trajectory.detach().view(-1, 4).data.numpy()
res = res > 0.5
print(np.sum(res.data.numpy() == desires.data.numpy()))
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(state_trajectory)

kmeans = KMeans(n_clusters=15).fit(state_trajectory)
X_centers = pca.transform(kmeans.cluster_centers_)
# np.save('trajectory/%d.npy' % tomita_type, kmeans.cluster_centers_)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.scatter(X_centers[:, 0], X_centers[:, 1])
plt.show()


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], color='blue', s=10, label='class 1')
ax.scatter(X_centers[:, 0], X_centers[:, 1], X_centers[:, 2], color='red', s=80, label='class 1')
plt.title('3D Scatter Plot')
plt.show()