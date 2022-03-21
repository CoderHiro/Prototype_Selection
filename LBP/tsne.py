from matplotlib import pyplot as plt
from scipy.sparse.construct import random
from sklearn import manifold
from sklearn.datasets import load_svmlight_file
from matplotlib import cm
import numpy as np


def tsne_image(all_data, all_label, save_path):
    label_list = ['Yuner.A', 'Lolyda.AA1', 'Instantaccess', 'Wintrim.BX', 'Allaple.A', 'Dialplatform.B', 'Dontovo.A', 'Skintrim.N', 'Rbot!gen', 'Swizzor.gen!I', 'Lolyda.AA2', 'Fakerean', 'VB.AT', 'C2LOP.gen!g', 'Agent.FYI', 'Autorun.K', 'Obfuscator.AD', 'Malex.gen!J', 'Swizzor.gen!E', 'Lolyda.AA3', 'Alueron.gen!J', 'Lolyda.AT', 'C2LOP.P', 'Adialer.C', 'Allaple.L']

    # label_list = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda','Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']
    color_list = cm.rainbow(np.linspace(0, 1, len(label_list)))
    color = []
    label = []
    for i in all_label:
        color.append(color_list[int(i)-1])
        label.append(label_list[int(i)-1])

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(all_data)

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=color, s=0.3)
    plt.grid(True)

    plt.savefig(save_path, format='eps')

data = load_svmlight_file("/home/cgh/prototype_selection/LBP/ft_output/Malimg_resnet50.txt")
save_path = './Malimg_resnet50.eps'
tsne_image(data[0].todense(), data[1], save_path)
