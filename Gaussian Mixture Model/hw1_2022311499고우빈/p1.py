import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

file_path = os.path.dirname(os.path.abspath(__file__))

def make_ellipses(gmm, ax):
    ax = plt.gca()
    for n, color in enumerate(np.unique(convertedTarget)):
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
            gmm.means_[n, :2], v[0], v[1], 180 + angle, color='b' if color in [1, 3, 4, 6, 9, 11, 12, 14] else 'r'
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

def convertTo16Classes(originTarget):
  for idx, i in enumerate(inputData):
    if 1 >= i[1] > 0.5:
      if -1<= i[0] <-0.5:
        originTarget[idx] = 0
      elif -0.5 <= i[0] < 0:
        originTarget[idx] = 1
      elif 0 <= i[0] < 0.5:
        originTarget[idx] = 2
      else:
        originTarget[idx] = 3
    elif 0.5 >= i[1] > 0:
      if -1<= i[0] <-0.5:
        originTarget[idx] = 4
      elif -0.5 <= i[0] < 0:
        originTarget[idx] = 5
      elif 0 <= i[0] < 0.5:
        originTarget[idx] = 6
      else:
        originTarget[idx] = 7
    elif 0 >= i[1] > -0.5:
      if -1<= i[0] <-0.5:
        originTarget[idx] = 8
      elif -0.5 <= i[0] < 0:
        originTarget[idx] = 9
      elif 0 <= i[0] < 0.5:
        originTarget[idx] = 10
      else:
        originTarget[idx] = 11
    else:
      if -1<= i[0] <-0.5:
        originTarget[idx] = 12
      elif -0.5 <= i[0] < 0:
        originTarget[idx] = 13
      elif 0 <= i[0] < 0.5:
        originTarget[idx] = 14
      else:
        originTarget[idx] = 15
  return originTarget

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

targetData = readTargetFile('./data/p1_train_target_win.txt')
inputData = readInputFile('./data/p1_train_input_win.txt')

convertedTarget = convertTo16Classes(targetData)
for i in (range(0, len(inputData))):
  plt.plot(float(inputData[i][0]), float(inputData[i][1]), 'b.' if convertedTarget[i] in [1, 3, 4, 6, 9, 11, 12, 14] else 'r.')

inputData = np.array(inputData)
targetData = np.array(targetData)
X = inputData[:, ::-1] # flip axes for better plotting
gmm = GaussianMixture(n_components=2, verbose=True, tol=0.0001, covariance_type='spherical', random_state=0).fit(inputData)
gmm.means_init = np.array(
  [inputData[targetData == i].mean(axis=0) for i in range(2)]
)
make_ellipses(gmm, plt)
y_train_pred = gmm.predict(inputData)
train_count = 0
for idx, i in enumerate(y_train_pred):
  if(i in [1, 3, 4, 6, 9, 11, 12, 14] and convertedTarget[idx] in [1, 3, 4, 6, 9, 11, 12, 14]):
    train_count += 1
  elif(i in [0, 2, 5, 7, 8, 10, 13, 15] and convertedTarget[idx] in [0, 2, 5, 7, 8, 10, 13, 15]):
    train_count += 1
train_accuracy = (train_count / 225) * 100
print("Train accuracy: %.1f" % train_accuracy)
# plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy)

test_input = readInputFile('./data/p1_test_input_win.txt')
test_target = readTargetFile('./data/p1_test_target_win.txt')
test_convertedTarget = convertTo16Classes(test_target)
y_test_pred = gmm.predict(test_input)
test_count = 0
for idx, i in enumerate(y_test_pred):
  if(i in [1, 3, 4, 6, 9, 11, 12, 14] and test_convertedTarget[idx] in [1, 3, 4, 6, 9, 11, 12, 14]):
    test_count += 1
  elif(i in [0, 2, 5, 7, 8, 10, 13, 15] and test_convertedTarget[idx] in [0, 2, 5, 7, 8, 10, 13, 15]):
    test_count += 1
test_accuracy = (test_count / 225) * 100
print("Test accuracy: %.1f" % test_accuracy)
# plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy)

# plt.scatter(inputData[:, 0], inputData[:, 1], c=convertedTarget)
# plot_gmm(gmm, inputData)
plt.show()
