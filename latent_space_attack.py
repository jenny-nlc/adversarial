import keras
import numpy as np
import seaborn as sns
from keras import backend as K
from matplotlib import pyplot as plt
from scipy.stats import norm

import src.utilities as U
from cleverhans import attacks
from cleverhans.model import CallableModelWrapper
from load_dropout_model import load_drop_model
from train_cdropout_3s_7s import define_cdropout_3s_7s, mnist_to_3s_and_7s
from train_mnist_vae import define_VAE
from scipy import optimize

plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 


def visualise_latent_space(decoder, n_grid=10):
    grid = norm.ppf(np.linspace(0.01,0.99, n_grid))

    xx, yy = np.meshgrid(grid, grid)

    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
    Z = decoder.predict(X)

    Z = Z.reshape(n_grid, n_grid, 28,28)

    imgrid = np.concatenate(
        [np.concatenate([Z[i,j] for i in range(n_grid)], axis=1)
         for j in range(n_grid)], axis=0)
    plt.imshow(imgrid, cmap='gray_r')

def get_models_3s_7s():
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights_3s_7s.h5')
    decoder.load_weights('save/dec_weights_3s_7s.h5')
    model = define_cdropout_3s_7s()
    model.load_weights('save/mnist_cdrop_3s_7s.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder

def get_models():
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights.h5')
    decoder.load_weights('save/dec_weights.h5')

    model = load_drop_model('save/mnist_cdrop_cnn_run_2.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder



class EnsembleWrapper:
    def __init__(self, modellist):
        self.ms = modellist
    def get_results(self, X):
        mc_preds = np.array([model1.predict(X) for _ in range(25)] + [model2.predict(X) for _ in range(25)])
        preds = mc_preds.mean(axis=0)
        ent = -np.sum(preds * np.log(preds + 1e-10), axis=-1)
        bald = ent - np.mean( - np.sum(mc_preds * np.log(mc_preds + 1e-10), axis=-1), axis=0)
        return preds, ent, bald

class LatentSpaceAttack:
    def __init__(self,
                 model,
                 decoder,
                 latent_dim=2,
                 percentile = 0.01,
                 r_latent_min = norm.ppf(0.999), #smallest distance from the origin to search
                 r_latent_max = 20, #arbitrary max distance
                 batch_size = 500):
        self.model = model
        self.latent_dim = latent_dim
        self.decoder = decoder
        self.percentile = 0.01
        self.batch_size = batch_size
        self.r_latent_min = r_latent_min
        self.r_latent_max = r_latent_max

    def generate_random(self,n_examples, batch_size = None):
       
        batch_size = self.batch_size if batch_size is None else batch_size 

        examples = []
        for i in range(n_examples):
        
            # choose random directions over the unit sphere by normalising
                # gaussian random vars
            rand_dirs = np.random.randn(batch_size, self.latent_dim)
            rand_dirs /= np.linalg.norm(rand_dirs, axis=1, ord=2)[:, None]

            #choose random magnitudes in the interval [min, max)
            mags = (self.r_latent_max - self.r_latent_min) * np.random.random(size=batch_size) + self.r_latent_min

            latent_samples = mags[:, None] * rand_dirs

            #evaluate the samples
            preds = model.predict(decoder.predict(latent_samples))

            #choose the example with the most confident prediction for any class 

            best_pred = preds.max(axis=1)
            most_confident = best_pred.argmax()

            examples.append(
                decoder.predict(latent_samples[most_confident][None, :])
            )

        return np.concatenate(examples, axis=0)
    def generate_diff_evolution(self, n_examples, batch_size = None):

        def objective(z):
            #cost of a point in latent space
            s = self.decoder.predict(z[None, :])
            p = self.model.predict(s)
            return 1 - np.max(p)
        im = optimize.differential_evolution(
            objective,
            [(self.r_latent_min, self.r_latent_max) for _ in range(self.latent_dim)],
            maxiter=10)
        return im
        
            
if __name__ == '__main__':

    model, encoder, decoder = get_models()
    
    x_train, y_train, x_test, y_test = U.get_mnist()

    lsa = LatentSpaceAttack(model, decoder, batch_size = 200)
    im = lsa.generate_random()
