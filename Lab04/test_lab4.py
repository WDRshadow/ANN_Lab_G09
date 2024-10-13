import unittest

from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

class TestRBM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRBM, self).__init__(*args, **kwargs)

        self.image_size = [28,28]
        train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=self.image_size, n_train=60000, n_test=10000)
            
        n_train_fraction = int(len(train_imgs) * 0.05)
        print("Smaller sample size, first:", n_train_fraction, "values")
        train_imgs = train_imgs[:n_train_fraction]
        np.random.shuffle(train_imgs)

        self.train_imgs = train_imgs

    def test_hidden_nodes(self):
        start = 200
        end = 500
        step = 100

        average_reconstruction_loss = []

        # RELEVANT VALUES TO MODIFY:
        n_iterations=5
        # TODO: STEP IS DEFINED IN THE RBM CODE, ADAPT ACCORDINGLY
        rbm_step = 2

        for i, ndim_hidden in enumerate(range(start, end + 1, step)):            
            rbm = RestrictedBoltzmannMachine(ndim_visible=self.image_size[0]*self.image_size[1],
                                             ndim_hidden=ndim_hidden,
                                             is_bottom=True,
                                             image_size=self.image_size,
                                             is_top=False,
                                             n_labels=10,
                                             batch_size=10
            )

            mean_reconst_loss = []    
            rbm.cd1(visible_trainset=self.train_imgs, n_iterations=n_iterations, plot_loss=mean_reconst_loss)
            average_reconstruction_loss.append(mean_reconst_loss)

        for i, ndim_hidden in enumerate(range(start, end + 1, step)):
            plt.plot(average_reconstruction_loss[i], label=f'{ndim_hidden} hidden units')
        
        iter_amm = int((n_iterations+1)/rbm_step)
        plt.xticks(range(iter_amm), labels=[str(i*rbm_step) for i in range(iter_amm)])

        plt.title('Reconstruction Loss vs Iterations for Different Hidden Unit Sizes')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reconstruction Loss')
        plt.legend()
        plt.show()