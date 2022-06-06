import os
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

path = os.path.abspath(__file__)

with open(os.path.dirname(path) + "/train.pkl", "rb") as fr:
  Xtrain = pickle.load(fr)

with open(os.path.dirname(path) + "/test.pkl", "rb") as fr:
  Xtest = pickle.load(fr)

# f = open(os.path.dirname(path) + "/a.txt", 'w')
# for row in data:
#   np.savetxt(f, row, fmt='%s')
# f.close()

with open(os.path.dirname(path) + "/Ytrain.pkl", "rb") as fr:
  Ytrain = pickle.load(fr)

with open(os.path.dirname(path) + "/Ytest.pkl", "rb") as fr:
  Ytest = pickle.load(fr)

clf = MLPClassifier(random_state=0, hidden_layer_sizes=(75,), activation='relu', solver='adam').fit(Xtrain[38999: ], Ytrain[38999: ])
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(Xtrain[38999: ], Ytrain[38999: ])
Y_pred = clf.predict(Xtest[5030: ])
print(Y_pred)

# for i, v in enumerate(Ytest[5030: ]):
#   if v != Y_pred[i]:
#     print("Label: " + str(v) + "   Pred: " + str(Y_pred[i]) )

# acc_train = clf.score(Xtrain, Ytrain)
# print(acc_train)
acc_test = clf.score(Xtest[5030: ], Ytest[5030: ])
print(acc_test)