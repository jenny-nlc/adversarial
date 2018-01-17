import keras
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm

import src.utilities as U
from train_cdropout_3s_7s import define_cdropout_3s_7s, mnist_to_3s_and_7s
from train_mnist_vae import define_VAE
from load_dropout_model import load_drop_model

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


def get_uncertainty_samples(mc_model,encoder, decoder, extent, n_grid=100):

    z_min, z_max = extent
    grid = np.linspace(z_min,z_max,n_grid)

    xx, yy = np.meshgrid(grid, grid)
    Z = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)

    #sample the image at this point in latent space, and get the BALD
    X = decoder.predict(Z) #produce corresponding images for the latent space grid
    preds,entropy, bald = mc_model.get_results(X)
    return preds, entropy.reshape(xx.shape), bald.reshape(xx.shape) 

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

    model = load_drop_model('save/mnist_cdrop_cnn.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder


if __name__ == '__main__':

    #mc_model, encoder, decoder = get_models_3s_7s()
    
    #x_train, y_train, x_test, y_test = mnist_to_3s_and_7s(U.get_mnist())

    mc_model, encoder, decoder = get_models()
    
    x_train, y_train, x_test, y_test = U.get_mnist()


    #visualise training data distribution

    proj_x_train = encoder.predict(x_train)


    zmin, zmax = -10,10
    preds, plot_ent, plot_bald = get_uncertainty_samples( mc_model, encoder, decoder, [zmin, zmax], n_grid=50)
    
    f, ax = plt.subplots(1,2)
    ax[0].scatter(proj_x_train[:,0],
                proj_x_train[:,1],
                c = y_train.argmax(axis=1),
                marker=',',
                s=1,
                cmap='Set3'
    )

    extent = [zmin, zmax, zmin, zmax] 
    im = ax[0].imshow(plot_bald,
                      cmap='gray',
                      origin='lower',
                      alpha=0.9,
                      extent=extent,
                     )
    ax[0].set_xlabel('First Latent Dimension')
    ax[0].set_ylabel('Second Latent Dimension')
    ax[0].set_title('BALD of Concrete Dropout Model')

    latent_z1, latent_z2 = 0,0 #starting position
    proj = ax[1].imshow(decoder.predict(np.array([[latent_z1, latent_z2]])).squeeze(), cmap='gray_r')

    last_sample = None
    def on_click(click):
        global last_sample

        if click.xdata != None and click.ydata != None and click.inaxes==ax[0]:
            z1 = click.xdata
            z2 = click.ydata
            dream = decoder.predict(np.array([[z1, z2]]))
            pred,entropy,bald = mc_model.get_results(dream)
            print("Predicted Class: {}, prob: {}".format(pred.argmax(axis=1), pred.max(axis=1)))
            print("Predictive Entropy: {}".format(entropy[0]))
            print("BALD Score:         {}".format(bald[0]))
            proj.set_data(dream.squeeze())
            plt.draw()
            last_sample = dream
    f.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
