import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import manifold


def tsne_value(data):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)
    return data_tsne

def tsne_image(data, true_label, predict_label, save_path):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

    scatter0 = ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], c=true_label, s=0.3)
    legend0 = ax1.legend(*scatter0.legend_elements(), title="Classes")
    ax1.add_artist(legend0)
    ax1.set_title('real_labels')
    ax1.grid(True)

    scatter1 = ax2.scatter(data_tsne[:, 0], data_tsne[:, 1], c=predict_label, s=0.3)
    legend1 = ax2.legend(*scatter1.legend_elements(), title="Classes")
    ax2.add_artist(legend1)
    ax2.set_title('predict_labels')
    ax2.grid(True)

    fig.tight_layout()

    plt.savefig(save_path)

def tsne_3D_image(data, true_label, predict_label, save_path):

    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)

    fig = plt.figure(figsize=(10,6)) 
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    scatter0 = ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], data_tsne[:, 2], c=true_label, s=0.3)
    legend0 = ax1.legend(*scatter0.legend_elements(), title="Classes")
    ax1.add_artist(legend0)
    ax1.set_title('real_labels')
    ax1.grid(True)

    scatter1 = ax2.scatter(data_tsne[:, 0], data_tsne[:, 1], data_tsne[:, 2], c=predict_label, s=0.3)
    legend1 = ax2.legend(*scatter1.legend_elements(), title="Classes")
    ax2.add_artist(legend1)
    ax2.set_title('predict_labels')
    ax2.grid(True)

    fig.tight_layout()

    plt.savefig(save_path)