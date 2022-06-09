from cProfile import label
import os

from sklearn.mixture import GaussianMixture


file_path = os.path.dirname(os.path.abspath(__file__))

originData = open(file_path + '/train_input.txt')
originLabel = open(file_path + '/train_target.txt')
dataLines = originData.readlines()
labelLines = originLabel.readlines()
data0 = []
data1 = []
data_whole = []
label = []

for i in labelLines:
  label.append(int(i))

for idx, i in enumerate(dataLines):
  tmp = i.split(',')
  debt1 = int(tmp[10]) - int(tmp[16])
  debt2 = int(tmp[9]) - int(tmp[15])
  debt3 = int(tmp[8]) - int(tmp[14])
  debt4 = int(tmp[7]) - int(tmp[13])
  debt5 = int(tmp[6]) - int(tmp[12])
  debt6 = int(tmp[5]) - int(tmp[11])
  debt_avg = (debt1*6 + debt2*5 + debt3*4 + debt4*3 + debt5*1 + debt6) / 6
  # data.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), debt1, debt1+debt2, debt1+debt2+debt3, debt1+debt2+debt3+debt4, debt1+debt2+debt3+debt4+debt5, debt1+debt2+debt3+debt4+debt5+debt6])
  if(label[idx] == 0):
    data0.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), debt_avg])
  else:
    data1.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), debt_avg])
  data_whole.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), debt_avg])




# print(label)
# for idx, i in enumerate(data):
#   if(i[5] < 0):
#     print(idx, i)

gmm0 = GaussianMixture(n_components=1, verbose=True, tol=1, covariance_type='tied', random_state=0).fit(data0)
gmm1 = GaussianMixture(n_components=1, verbose=True, tol=1, covariance_type='tied', random_state=0).fit(data1)

prob0 = gmm0.predict_proba(data_whole)
prob1 = gmm0.predict_proba(data_whole)

for i in prob0:
  if i[0] != 1:
    print(i)