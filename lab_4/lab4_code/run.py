from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=500,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=20
    # )
    # rbm.cd1(visible_trainset=train_imgs, n_iterations=15)#15

    # #FOR DEEP NETWORK, INCLUDE MORE LAYERS
    # rbm1 = RestrictedBoltzmannMachine(ndim_visible=784,
    #                                  ndim_hidden=500,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=20
    # )
    # rbm1.cd1(visible_trainset=train_imgs, n_iterations=10)#15

    # p_h1, h1 = rbm1.get_h_given_v(train_imgs)



    # rbm2 = RestrictedBoltzmannMachine(ndim_visible=500,
    #                                  ndim_hidden=500,
    #                                  is_bottom=False,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=20
    # )
    # rbm2.cd1(visible_trainset=p_h1, n_iterations=10)#15

    # pass

    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''
    print("\nGreedy layer-wise training..")
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=5000)#1000
    
    print("\nRecognition after greedy layer-wise training..")
    # dbn.recognize(train_imgs, train_lbls)    
    steps_tr, acc_tr, conf_tr = dbn.recognize_trace(train_imgs, train_lbls)

    
    print("\nRecognition on test set after greedy layer-wise training..")
    # dbn.recognize(test_imgs, test_lbls)
    steps_te, acc_te, conf_te = dbn.recognize_trace(test_imgs, test_lbls)

    # Accuracy overlay
    plt.figure(figsize=(6,4))
    plt.plot(steps_tr, acc_tr * 100, label="train")
    plt.plot(steps_te, acc_te * 100, label="test")
    plt.xlabel("Topmost RBM Gibbs sampling step")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Convergence over Gibbs Sampling Steps")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task42.accuracy_train_vs_test.png")
    plt.close()

    # Confidence overlay
    plt.figure(figsize=(6,4))
    plt.plot(steps_tr, conf_tr, label="train")
    plt.plot(steps_te, conf_te, label="test")
    plt.xlabel("Topmost RBM Gibbs sampling step")
    plt.ylabel("Mean max label probability")
    plt.title("Label 'Confidence' over Gibbs Sampling Steps")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("task42.confidence_train_vs_test.png")
    plt.close()


    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''
    # print("\nFine-tune wake-sleep training..")
    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1000)#1000

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
