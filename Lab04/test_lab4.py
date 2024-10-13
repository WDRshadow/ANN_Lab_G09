import unittest

from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

class TestRBM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRBM, self).__init__(*args, **kwargs)

        self.image_size = [28,28]
        train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=self.image_size, n_train=60000, n_test=10000)
            
        n_train_fraction = int(len(train_imgs) * 0.1)
        print("Smaller sample size, first:", n_train_fraction, "values")
        train_imgs = train_imgs[:n_train_fraction]
        np.random.shuffle(train_imgs)

        self.train_imgs = train_imgs
        self.test_imgs = test_imgs

        # RELEVANT VALUES TO MODIFY:
        self.n_iterations=101
        # TODO: MUST BE EQUAL TO STEP DEFINED IN THE RBM CODE, ADAPT ACCORDINGLY
        self.rbm_step = 2

    def test_hidden_nodes(self):
        return
        start = 200
        end = 500
        step = 100

        average_reconstruction_loss = []

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
            rbm.cd1(visible_trainset=self.train_imgs, n_iterations=self.n_iterations, plot_loss=mean_reconst_loss)
            average_reconstruction_loss.append(mean_reconst_loss)

        for i, ndim_hidden in enumerate(range(start, end + 1, step)):
            plt.plot(average_reconstruction_loss[i], label=f'{ndim_hidden} hidden units')
        
        iter_amm = int((self.n_iterations+1)/self.rbm_step)
        plt.xticks(range(iter_amm), labels=[str(i*self.rbm_step) for i in range(iter_amm)])

        plt.title('Reconstruction Loss vs Iterations for Different Hidden Unit Sizes')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reconstruction Loss')
        plt.legend()
        plt.show()


    def test_weight_and_bias(self):
        weight_changes = []
        visible_bias_changes = []
        hidden_bias_changes = []

        rbm = RestrictedBoltzmannMachine(ndim_visible=self.image_size[0]*self.image_size[1],
                                         ndim_hidden=500,
                                         is_bottom=True,
                                         image_size=self.image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=20,
                                         weight_changes=weight_changes,
                                         visible_bias_changes=visible_bias_changes,
                                         hidden_bias_changes=hidden_bias_changes
        )

        mean_reconst_loss = []    
        rbm.cd1(visible_trainset=self.train_imgs, n_iterations=self.n_iterations, plot_loss=mean_reconst_loss)

        plt.plot(weight_changes, label="Weight changes")
        plt.plot(visible_bias_changes, label="Visible bias changes")
        plt.plot(hidden_bias_changes, label="Hidden bias changes")
        
        iter_amm = int((self.n_iterations+1)/self.rbm_step)
        plt.xticks(range(iter_amm), labels=[str(i*self.rbm_step) for i in range(iter_amm)])

        plt.title('Convergence of Weights and Biases')
        plt.xlabel('Epoch')
        plt.ylabel('Update Magnitude (Norm)')
        plt.legend()
        plt.show()

        for i in range(5):
            reconstructed_img = reconstruct_image(rbm, self.test_imgs[i])
            visualize_reconstruction(self.test_imgs[i].reshape(rbm.image_size), reconstructed_img)
    