import numpy as np
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rcParams['mathtext.fontset'] = 'cm'

positive_points = np.array([[0.8, 0.4], [0.74, 0.8], [0.5, 0.75], [0.1, 0.58], [0.2, 0.9]])
negative_points = np.array([[0.73, 0.23], [0.53, 0.12], [0.2, 0.3], [0.4, 0.45], [0.94, 0.7]])

positive_annotations = [1, 7, 8, 9, 10]
negative_annotations = [2, 3, 4, 5, 6]

plt.close()
plt.figure(figsize=(6,6))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(positive_points[:, 0], positive_points[:, 1], 130, marker='x', c='red')
plt.scatter(negative_points[:, 0], negative_points[:, 1], 220, marker='.', c='blue')
for i in range(0, len(positive_annotations)):
    plt.text(positive_points[i, 0] + 0.02, positive_points[i, 1] + 0.005, positive_annotations[i], fontsize=16)
for i in range(0, len(negative_annotations)):
    plt.text(negative_points[i, 0] + 0.02, negative_points[i, 1] + 0.005, negative_annotations[i], fontsize=16)
plt.savefig('Adaboost_data.pdf')
