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
               burn_in=100,
               sample_every=10,
               verbose = False
):
    
    #assume x is a 1-d vector for now

    E, jac = obj(x)  # energy, jacobian

    samples = []
    losses = []
    tau_lo = np.int( 0.5 * tau)
    tau_hi = np.int(1.5 * tau)

    ep_lo = 0.8 * epsilon
    ep_hi = 1.2 * epsilon
    i = 0
    while( len(samples) < n_samps):

        #sample tau and epsilon
        tau = np.random.randint(tau_lo, high=tau_hi)
        ep  = ep_lo + np.random.random() * (ep_hi - ep_lo)
        p = np.random.randn(*x.shape)
        H = 0.5 * p @ p.T + E

        x_prop = x.copy(); jac_prop = jac.copy();


        for t in range(tau): #tau leapfrog steps
            p      = p - 0.5 * ep * jac_prop  # half step in p
            x_prop = x_prop + ep * p          # update x

            E_prop, jac_prop = obj(x_prop)    # update grad
            
            p      = p - 0.5 * ep * jac_prop  # step p again

        H_prop = 0.5 * p @ p.T + E_prop       # Hamiltonian at proposal point

        dH = H_prop - H
        # Metropolis Hastings step to correct for the discrete dynamics
        if dH < 0 or np.random.random() < np.exp(-dH):
            # accept the step
            x = x_prop.copy()
            jac = jac_prop.copy()
            E = E_prop
        else:
            pass
        
        losses.append(E)
        if i > burn_in and i % sample_every == 0:
            samples.append(x)
        i += 1
        if verbose:
            print('iter:',i, 'Energy:', E, 'grad:', jac, 'x:', x)
    return np.array(samples).squeeze(), np.array(losses).squeeze()

    
    


if __name__ == '__main__':
    _, _, decoder = g_mmnist.define_CVAE()
    #load model
    decoder.load_weights('save/mmnist_dec_weights_latent_dim_2.h5')
   
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
            ims = np.broadcast_to(image, (z.shape[0], 28, 28, 1))
            nll, dnll_dz = get_loss_and_grads([z, c, ims])
            #use linearity of differentiation to get the gradient
            return nll + nlogprior, dnll_dz + d_nlogprior_dz
        
        return obj

    def eval_probabilities(image, n_samps=64, burn_in=20, epsilon=7e-2, tau=80, sample_every=1):
        logprobs = np.zeros(10)

        for c in range(10):
            obj = make_objective(image, c)
            samples, losses = hmc_sample(np.random.randn(1,2),
                                        obj,
                                        tau=tau,
                                        epsilon = epsilon,
                                        n_samps=n_samps,
                                        burn_in=burn_in,
                                        sample_every=sample_every)


            n_samps = samples.shape[0]
            samps1 = samples[:n_samps//2]
            samps2 = samples[n_samps//2:]
            density = GaussianMixture(n_components=1)
            density.fit(samps1)

            # evaluate the estimator of Kingma and Welling
            qz = density.score_samples(samps2) # log Probability of samples under the density model

            lE, _ = obj(samps2) # the (negative) log liklihood + log prior  
            logprobs[c] = - np.log((np.mean(np.exp(qz + lE))))
        logprob = np.log( np.exp( logprobs + np.log(0.1) ).mean())
        return logprob, logprobs
  
    mmnist = h5py.File('datasets/manifold_mnist.h5', 'r')

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
