import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 生成隨機數據
data = np.random.randn(1000).reshape(-1, 1)

# 應用高斯混合模型
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(data)
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)

# 可視化結果
plt.plot(x, pdf, label='GMM')
plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
plt.legend()
plt.show()