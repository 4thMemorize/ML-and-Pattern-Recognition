import os


file_path = os.path.dirname(os.path.abspath(__file__))

originData = open(file_path + './train_input.txt')
lines = originData.readlines()

data = []

for i in lines:
  tmp = i.split(',')
  debt1 = int(tmp[10]) - int(tmp[16])
  debt2 = int(tmp[9]) - int(tmp[15])
  debt3 = int(tmp[8]) - int(tmp[14])
  debt4 = int(tmp[7]) - int(tmp[13])
  debt5 = int(tmp[6]) - int(tmp[12])
  debt6 = int(tmp[5]) - int(tmp[11])
  data.append([int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), debt1, debt1+debt2, debt1+debt2+debt3, debt1+debt2+debt3+debt4, debt1+debt2+debt3+debt4+debt5, debt1+debt2+debt3+debt4+debt5+debt6])

print(data)