
import numpy as np
from sklearn.decomposition import PCA
import time

N = 1000
D = 152064
k = 4

print(f"Creating random data ({N}, {D})...")
X = np.random.randn(N, D).astype(np.float32)

print("Fitting PCA...")
st = time.time()
pca = PCA(n_components=k-1)
pca.fit(X)
print(f"PCA fit took {time.time() - st:.2f}s")
