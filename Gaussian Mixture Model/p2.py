import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

file_path = os.path.dirname(os.path.abspath(__file__))

def make_ellipses(gmm, ax):
    ax = plt.gca()
    for n, color in enumerate([0,1]):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], 180 + angle, color='b' if color == 0 else 'r'
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    print(labels)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def readTargetFile(path):
  targetFile = open(file_path+path, 'r')
  targetData = []
  lines = targetFile.readlines()
  for i in lines:
    tmp = i.split()
    # loop = tmp
    # for idx, j in enumerate(loop):
    #   tmp[idx] = float(j)
    #   print(tmp)
    targetData.append(int(tmp[0]))
  targetFile.close()
  return targetData

def readInputFile(path):
  inputFile = open(file_path+path, 'r')
  inputData = []
  lines = inputFile.readlines()
  for i in lines:
    tmp = i.split()
    loop = tmp
    for idx, j in enumerate(loop):
      tmp[idx] = float(j)
    inputData.append(tmp)
  inputFile.close()
  return inputData

targetData = readTargetFile('./data/p2_train_target_win.txt')
inputData = readInputFile('./data/p2_train_input_win.txt')

for i in (range(0, len(inputData))):
  plt.plot(float(inputData[i][0]), float(inputData[i][1]), 'b.' if targetData[i] == 0 else 'r.')

inputData = np.array(inputData)
targetData = np.array(targetData)
X = inputData[:, ::-1] # flip axes for better plotting
gmm = GaussianMixture(n_components=2, verbose=True, tol=0.0001, covariance_type='spherical', max_iter=50, random_state=0).fit(inputData)
gmm.means_init = np.array(
  [inputData[targetData == i].mean(axis=0) for i in range(2)]
)
make_ellipses(gmm, plt)
y_train_pred = gmm.predict(inputData)
train_count = 0
for idx, i in enumerate(y_train_pred):
  if(i == targetData[idx]):
    train_count += 1
train_accuracy = (train_count / 225) * 100
print("Train accuracy: %.1f" % train_accuracy)
# plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy)

test_input = readInputFile('./data/p2_test_input_win.txt')
test_target = readTargetFile('./data/p2_test_target_win.txt')
y_test_pred = gmm.predict(test_input)
test_count = 0
for idx, i in enumerate(y_test_pred):
  if(i == targetData[idx]):
    test_count += 1
test_accuracy = (test_count / 225) * 100
print("Test accuracy: %.1f" % test_accuracy)
# plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy)

# plt.scatter(inputData[:, 0], inputData[:, 1], c=convertedTarget)
# plot_gmm(gmm, inputData)
plt.show()
