import h5py
import keras
import numpy as np
from keras import backend as K
import generate_manifold_mnist as g_mmnist
from sklearn.mixture import GaussianMixture

from matplotlib import pyplot as plt

def binary_crossentropy(y, y_):
    return y * np.log(y) + (1 - y) * np.log(1 - y_)

def hmc_sample(x,  
               obj, 
               tau=100,
               epsilon=1e-2,
               n_samps=100,
               burn_in=20,
               sample_every=2,
               verbose = False):
    
    # work on batches; x is a N by d tensor. 

    E, jac = obj(x)  # energy, jacobian

    samples = np.zeros((x.shape[0], n_samps, x.shape[1])) #batch_size x samples x d
    sample_energies = np.zeros((x.shape[0], n_samps)) #batch_size x samples x d
    losses = []

    tau_lo = np.int( 0.5 * tau)
    tau_hi = np.int(1.5 * tau)

    ep_lo = 0.8 * epsilon
    ep_hi = 1.2 * epsilon
    i = 0
    sampind = 0
    while( sampind < n_samps):

        #sample tau and epsilon
        tau = np.random.randint(tau_lo, high=tau_hi)
        ep  = ep_lo + np.random.random() * (ep_hi - ep_lo)

        p = np.random.randn(*x.shape)
        H = 0.5 * (p **2).sum(axis=1) + E

        x_prop = x.copy(); jac_prop = jac.copy();


        for t in range(tau): #tau leapfrog steps
            p      = p - 0.5 * ep * jac_prop  # half step in p
            x_prop = x_prop + ep * p          # update x

            E_prop, jac_prop = obj(x_prop)    # update grad
            
            p      = p - 0.5 * ep * jac_prop  # step p again

        H_prop = 0.5 * (p ** 2).sum(axis=1) + E_prop       # Hamiltonian at proposal point

        dH = H_prop - H
        # Metropolis Hastings step to correct for the discrete dynamics
        to_update = np.logical_or(dH < 0, np.random.random(dH.shape) < np.exp(-dH))
        x[to_update]   = x_prop[to_update]
        jac[to_update] = x_prop[to_update]
        E[to_update]   = E_prop[to_update]
        losses.append(E.copy())
        if i > burn_in and i % sample_every == 0:
            samples[:, sampind, :] = x
            sample_energies[:, sampind] = E
            sampind += 1
        i += 1
        if verbose:
            print('iter:',i, 'Energy:', E, 'grad:', jac, 'x:', x)
    return samples, np.array(losses).squeeze().T, sample_energies

_, _, decoder = g_mmnist.define_CVAE()
#load model
decoder.load_weights('manifold_mnist/mmnist_dec_weights_latent_dim_2.h5')

def make_objective(image, class_target):
    # get gradient from keras model
    z_t, class_t = decoder.inputs
    prob_t = K.batch_flatten(decoder.outputs[0])
    image_t = K.placeholder(shape=(None, 28, 28, 1))
    x_t = K.batch_flatten(image_t)

    negloglik = K.sum(K.binary_crossentropy(x_t, prob_t), axis=-1) # log liklihood of x

    grad = K.gradients([negloglik], [z_t])[0]

    get_loss_and_grads = K.function([z_t, class_t, image_t], [negloglik, grad] )

    def obj(z):
        # log N(0,1) prior, trivial analytical form
        nlogprior      = np.log(np.sqrt( 2 * np.pi)) +  .5 * (z ** 2).sum(axis=1)
        d_nlogprior_dz =  z
        c = keras.utils.to_categorical(class_target, num_classes=10)
        c = np.broadcast_to(c, (z.shape[0], c.shape[1]))
        nll, dnll_dz = get_loss_and_grads([z, c, image])
        #use linearity of differentiation to get the gradient
        return nll + nlogprior, dnll_dz + d_nlogprior_dz
    
    return obj

def eval_probabilities(images, n_samps=64, burn_in=20, epsilon=7e-2, tau=80, sample_every=1):
    logprobs = np.zeros((images.shape[0], 10))

    for c in range(10):
        print('class', c)
        obj = make_objective(images, c)
        samples, losses, sample_logliks = hmc_sample(np.random.randn(images.shape[0],2),
                                    obj,
                                    tau=tau,
                                    epsilon = epsilon,
                                    n_samps=n_samps,
                                    burn_in=burn_in,
                                    sample_every=sample_every)


        samps1 = samples[:, 0:n_samps//2]
        samps2 = samples[:, n_samps//2:]
        for i in range(images.shape[0]):
            density = GaussianMixture(n_components=1)
            density.fit(samps1[i])

            # evaluate the estimator of Kingma and Welling
            qz = density.score_samples(samps2[i]) # log Probability of samples under the density model
            lE = sample_logliks[i, n_samps//2:] #negative log likelihood of the samples
            logprobs[i,c] = - np.log((np.mean(np.exp(qz + lE))))
    logprob = np.log( np.exp( logprobs + np.log(0.1) ).mean(axis=1))
    return logprob, logprobs




if __name__ == '__main__':
     
    mmnist = h5py.File('manifold_mnist/manifold_mnist.h5', 'r')

    log_prob = []
    log_prob_classes = []
    # define the energy of the posteior
    for i in range(50):
        print('image: ',i)
        image = mmnist['x_train'][i]
        lp, lps = eval_probabilities(image)
        log_prob.append(lp)
        log_prob_classes.append(lps)
        
    # xx, yy = np.meshgrid(np.linspace(-5,5,100), np.linspace(-5,5,100))
    # X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)

    # pdf = np.exp(density.score_samples(X))
    # plt.contourf(xx,yy,pdf.reshape(xx.shape))
    # plt.scatter(samples[:,0], samples[:,1])
