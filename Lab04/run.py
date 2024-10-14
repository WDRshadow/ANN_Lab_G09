from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    n_train_fraction = int(len(train_imgs) * 1)
    n_test_fraction = int(len(test_imgs) * 1)
    print("Smaller sample size, first:", n_train_fraction, "values")
    train_imgs = train_imgs[:n_train_fraction]
    train_lbls = train_lbls[:n_train_fraction]
    test_imgs = test_imgs[:n_test_fraction]
    test_lbls = test_lbls[:n_test_fraction]

    # ''' restricted boltzmann machine '''
    #
    # print ("\nStarting a Restricted Boltzmann Machine..")
    #
    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=200,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=10
    # )
    #
    # rbm.cd1(visible_trainset=train_imgs, n_iterations=1000)
    #
    # for i in range(5):
    #     reconstructed_img = reconstruct_image(rbm, test_imgs[i])
    #     visualize_reconstruction(test_imgs[i].reshape(rbm.image_size), reconstructed_img)
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1000)

    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    #
    # ''' fine-tune wake-sleep training '''
    #
    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")