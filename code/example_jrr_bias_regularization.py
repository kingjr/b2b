from scipy.stats import ttest_1samp
from synthetic_v_1_0 import synthetic_data_v_1_0
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
dim_x = 100
dim_y = 100
n_samples = 1000
X, Y, E = synthetic_data_v_1_0(n_samples, snr=1, dim_x=dim_x, dim_y=dim_y)


alphas = np.logspace(-6, 6, 20)
ridge = RidgeCV(alphas=alphas, fit_intercept=False)
ols = LinearRegression(fit_intercept=False)

# cv
set1 = range(n_samples//2)
set2 = range(n_samples//2, n_samples)
cv = ShuffleSplit(100, test_size=.5, random_state=0)

# standard JRR Bias
E_hats = list()
for set1, set2 in cv.split(X, Y):
    # decode
    G = ridge.fit(Y[set1], X[set1])
    X_hat = G.predict(Y)
    # Encode
    # change this ridge by non regularized least square to avoid final bias
    E_hat = ridge.fit(X[set2], X_hat[set2]).coef_
    E_hats.append(E_hat)
E_hat = np.mean(E_hats, 0)

plt.fill_between(range(dim_x), np.diag(E), label='E')
plt.fill_between(range(dim_x), np.diag(E_hat), label='E_hat')
plt.legend()


selected = np.diag(E) == 1
t, p_val = ttest_1samp(np.diag(E_hat)[selected == 0], 0)
print('not selected feature are above chance: p_value=%.4f' % p_val)
plt.show()
