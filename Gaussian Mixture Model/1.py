import numpy as np
from matplotlib import pyplot as plt

targetFile = open('./data/p1_train_target_win.txt', 'r')
targetData = []
lines = targetFile.readlines()
for i in lines:
  tmp = i.split()
  # loop = tmp
  # for idx, j in enumerate(loop):
  #   tmp[idx] = float(j)
  #   print(tmp)
  targetData.append(tmp)
print(targetData)
targetFile.close()

inputFile = open('./data/p1_train_input_win.txt', 'r')
inputData = []
lines = inputFile.readlines()
for i in lines:
  tmp = i.split()
  # loop = tmp
  # for idx, j in enumerate(loop):
  #   tmp[idx] = float(j)
  #   print(tmp)
  inputData.append(tmp)
print(inputData)
inputFile.close()

for i in (range(0, len(inputData))):
  plt.plot(float(inputData[i][0]), float(inputData[i][1]), 'b.' if int(targetData[i][0]) == 0 else 'r.')
plt.show()