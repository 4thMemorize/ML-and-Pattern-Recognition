import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def estimateGaussian(X):
    m = X.shape[0]
    #compute mean of X
    sum_ = np.sum(X,axis=0)
    mu = (sum_/m)
    # compute variance of X
    var = np.var(X,axis=0)
    # print(mu, var)
    return mu,var

def multivariateGaussian(X, mu, sigma):
    k = len(mu)
    sigma=np.diag(sigma)
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma) * X,axis=1))
    return p



### Basic Plot Show ###
def showPlots(X):
  plt.scatter(X[:,0],X[:,1],marker="x")
  plt.show()

### Plot With Density ###
def plotDensity(X):
  mu, sigma = estimateGaussian(X)
  p = multivariateGaussian(X, mu, sigma)
  plt.figure(figsize=(8,6))
  plt.scatter(X[:,0],X[:,1],marker="x",c=p,cmap='viridis')
  plt.colorbar()
  plt.show()

### Density Estimation ###
def estimateDensity(X_train, n_components, covType):
  clf = mixture.GaussianMixture(n_components=n_components, covariance_type=covType)
  clf.fit(X_train)
  x = np.linspace(0.0, 20.0)
  y = np.linspace(-17.0, 10.0)
  X, Y = np.meshgrid(x, y)
  XX = np.array([X.ravel(), Y.ravel()]).T
  Z = -clf.score_samples(XX)
  Z = Z.reshape(X.shape)
  # CS = plt.contour(
  #     X, Y, Z, norm=LogNorm(vmin=10.0, vmax=20.0), levels=np.logspace(0, 2, 30)
  # )
  CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
)
  p = clf.predict_proba(X_train)
  print(p)
  plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", cmap='viridis')
  CB = plt.colorbar(CS, shrink=0.8, extend="both")
  plt.title("Negative log-likelihood predicted by a GMM")
  # plt.axis("tight")
  # plt.xlim([-20, 20])
  # plt.ylim([-20, 20])
  plt.show()

### Detect abnormal data ###
def detectAnomaly(X, epsilon):
  mu, sigma = estimateGaussian(X)
  p = multivariateGaussian(X, mu, sigma)
  plt.figure(figsize=(8,6))
  plt.scatter(X[:,0],X[:,1],marker="x",c=p,cmap='viridis')
  outliers = np.nonzero(p<epsilon)[0]
  plt.scatter(X[outliers,0],X[outliers,1],marker="o",facecolor="none",edgecolor="r",s=70)
  plt.colorbar()
  plt.show()


######################################################

# X, y_true = make_blobs(n_samples=500, centers=1, cluster_std=0.60, random_state=5)
# X_append, y_true_append = make_blobs(n_samples=20,centers=1, cluster_std=5,random_state=5)
# X = np.vstack([X,X_append])
# y_true = np.hstack([y_true, [1 for _ in y_true_append]])
# X = X[:, ::-1]

X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=0.60, random_state=1)
X_append, y_true_append = make_blobs(n_samples=50,centers=5, cluster_std=5,random_state=1)
X = np.vstack([X,X_append])
y_true = np.hstack([[0 for _ in y_true], [1 for _ in y_true_append]])
X = X[:, ::-1] # flip axes for better plotting
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=1, shuffle=True)

# showPlots(X)
# plotDensity(X)
# estimateDensity(X, 5, 'tied')
detectAnomaly(X, 0.001)

######################################################
