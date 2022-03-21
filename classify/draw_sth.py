import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.family'] = 'serif'

x = [i for i in range(10,100,10)]
y_vgg = [91.083, 94.022, 95.094, 95.702, 96.439, 96.649, 96.553, 96.856, 96.928]
y_resnet = [92.27,93.84,95.354,95.707,96.464,96.403,96.741,96.835,97.823,]
y_inception = [91.26,94.15,94.809,96.007,96.096,96.742,97.021,97.021,97.113]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y_vgg, color='tomato', marker='^')
ax.plot(x, y_resnet, color='royalblue', marker='.')
ax.plot(x, y_inception, color='orange', marker='*')
ax.grid(True)
plt.ylim(90,100)
plt.xlabel('Selected rate')
plt.ylabel('F1-score')
plt.legend(['VGG16', 'Resnet50', 'InceptionV3'], loc='upper right', 
          bbox_to_anchor=(0.35,  # horizontal
                          1.0),# vertical 
          ncol=1, fancybox=True)
plt.xticks(np.arange(10,110,10))
# plt.savefig('BIG15_vgg16_10-100.png')
plt.savefig('BIG15_vgg16_10-100.eps', format='eps')