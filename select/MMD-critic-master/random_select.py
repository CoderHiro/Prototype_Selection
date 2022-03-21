import os
import csv
import numpy as np

seed = 3

dataset = 'BIG15'
np.random.seed(seed)
# big15-9772 malimg-8406
test_index = np.random.randint(9772, size=8794)
test_index.sort()

all_name = []
with open("/home/cgh/prototype_selection/LBP/ft_output/{}_selected_name_list.csv".format(dataset), 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        all_name.append(line[0])
with open("/home/cgh/prototype_selection/select/MMD-critic-master/ft_output_90/{}_random_{}.csv".format(dataset, seed), 'w') as f:
    writer = csv.writer(f)
    for index in test_index:
        name = all_name[index]
        writer.writerow([name])