import numpy as np
import matplotlib.pyplot as plt
import json
import csv

dataset = 'Malimg'
model_type = 'vgg16'
selected_rate = 10

matrixs = 'f1-score'

if dataset == 'BIG15':
    x = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda','Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']
    class_num = 9
else:
    # x = ['Yuner.A', 'Lolyda.AA1', 'Instantaccess', 'Wintrim.BX', 'Allaple.A', 'Dialplatform.B', 'Dontovo.A', 'Skintrim.N', 'Rbot!gen', 'Swizzor.gen!I', 'Lolyda.AA2', 'Fakerean', 'VB.AT', 'C2LOP.gen!g', 'Agent.FYI', 'Autorun.K', 'Obfuscator.AD', 'Malex.gen!J', 'Swizzor.gen!E', 'Lolyda.AA3', 'Alueron.gen!J', 'Lolyda.AT', 'C2LOP.P', 'Adialer.C', 'Allaple.L']
    x = ['Yuner.A', 'Lolyda.AA1', 'Instantaccess', 'Wintrim.BX', 'Allaple.A', 'Dialplatform.B', 'Dontovo.A', 'Skintrim.N', 'Rbot!gen', 'Swizzor.gen!I', 'Lolyda.AA2', 'Fakerean', 'VB.AT', 'C2LOP.gen!g', 'Agent.FYI', 'Obfuscator.AD', 'Malex.gen!J', 'Swizzor.gen!E', 'Lolyda.AA3', 'Alueron.gen!J', 'Lolyda.AT', 'C2LOP.P', 'Adialer.C', 'Allaple.L']
    class_num = 25

with open('./result_analyse/{}_{}_all.json'.format(dataset, model_type), 'r') as f:
    reader = json.load(f)
    y_all = [reader[classes][matrixs] for classes in x]
    print(y_all)

with open('./result_analyse/{}_json/{}_{}_vgg16.json'.format(selected_rate, dataset, model_type), 'r') as f:
    reader = json.load(f)
    y_vgg16 = [reader[classes][matrixs] for classes in x]
    print(y_vgg16)

with open('./result_analyse/{}_json/{}_{}_resnet50.json'.format(selected_rate, dataset, model_type), 'r') as f:
    reader = json.load(f)
    y_resnet50 = [reader[classes][matrixs] for classes in x]
    print(y_resnet50)

with open('./result_analyse/{}_json/{}_{}_inceptionV3.json'.format(selected_rate, dataset, model_type), 'r') as f:
    reader = json.load(f)
    y_inceptionV3 = [reader[classes][matrixs] for classes in x]
    print(y_inceptionV3)

y_num = [0 for i in range(class_num)]
with open('./data/{}_labels.csv'.format(dataset), 'r') as f:
    reader = csv.reader(f)
    reader.__next__()
    for line in reader:
        y_num[int(line[1])-1] += 1
del(y_num[15])

plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(1, 1, figsize=(16,7))

ax.bar(x, y_num, color='coral', edgecolor='black', hatch='/')
ax.set_ylabel('Number of each family', fontsize='large')
plt.xticks(rotation=30)
plt.legend(['Number'], loc='upper left', 
          bbox_to_anchor=(0,  # horizontal
                          1.1),# vertical 
          ncol=3, fancybox=True)

ax2 = ax.twinx()
ax2.plot(x, y_all, linestyle='-', color='tomato', marker='x')
ax2.plot(x, y_vgg16, linestyle='-', color='gold', marker='o')
ax2.plot(x, y_resnet50, linestyle='-', color='cyan', marker='v')
ax2.plot(x, y_inceptionV3, linestyle='-', color='navy', marker='^')
ax2.set_ylabel('F1-score of each family', fontsize='large')
plt.legend(['All', 'VGG16', 'Resnet50', 'InceptionV3'], loc='upper right', 
          bbox_to_anchor=(1.0,  # horizontal
                          1.1),# vertical 
          ncol=4, fancybox=True)
plt.xticks(rotation=30)
plt.yticks(np.arange(0.0,1.2,0.2))

plt.savefig('./result_analyse/pngs/{}_{}_{}.eps'.format(selected_rate, dataset, model_type), format='eps')