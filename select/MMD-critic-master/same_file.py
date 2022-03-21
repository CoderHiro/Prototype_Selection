import csv
import os

dataset = "Malimg"
vgg16_path = "/home/cgh/prototype_selection/select/MMD-critic-master/selected_10/{}_vgg16_global.csv".format(dataset)
resnet_path = "/home/cgh/prototype_selection/select/MMD-critic-master/selected_10/{}_resnet50_global.csv".format(dataset)
inception_path = "/home/cgh/prototype_selection/select/MMD-critic-master/selected_10/{}_inceptionV3_global.csv".format(dataset)

vgg16_list = []
resnet_list = []
inception_list = []

with open(vgg16_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        vgg16_list.append(line[0])

with open(resnet_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        resnet_list.append(line[0])

with open(inception_path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        inception_list.append(line[0])

dict_for_list = {}
count = 0

for file_name in vgg16_list:
    if file_name in dict_for_list.keys():
        dict_for_list[file_name] += 1
    else:
        dict_for_list[file_name] = 1

for file_name in resnet_list:
    if file_name in dict_for_list.keys():
        dict_for_list[file_name] += 1
    else:
        dict_for_list[file_name] = 1

for file_name in inception_list:
    if file_name in dict_for_list.keys():
        dict_for_list[file_name] += 1
    else:
        dict_for_list[file_name] = 1

for file_name in dict_for_list.keys():
    if dict_for_list[file_name] >= 2:
        count += 1

print("Important file: {}".format(count))
print("All file: {}".format(len(dict_for_list)))