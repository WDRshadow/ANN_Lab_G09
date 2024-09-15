import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def graph_matrix(items, n1, n2):
    accuracy_matrix = np.array(items).reshape(len(n1), len(n2))

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(accuracy_matrix, annot=True, cmap="YlGnBu", xticklabels=n2, yticklabels=n1, fmt='.6f')
    ax.invert_yaxis()

    ax.set_xlabel('n2 (Number of Nodes in Second Layer)')
    ax.set_ylabel('n1 (Number of Nodes in First Layer)')
    ax.set_title('Accuracy for Different Node Combinations (n1 vs n2)')
    
    plt.show()
