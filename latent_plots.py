import os

import keras
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
from scipy.stats import norm

import src.utilities as U
from cleverhans import attacks
from cleverhans.model import CallableModelWrapper
from load_dropout_model import load_drop_model
from train_cdropout_3s_7s import define_cdropout_3s_7s, mnist_to_3s_and_7s
from train_mnist_vae import define_VAE
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
    K.set_learning_phase(True)
    model = keras.models.load_model('save/mnist_cnn_run_1.h5')
    mc_model = U.MCModel(model, model.input, n_mc=50)
    #we have been using more mc samples elsewhere, but save time for now
    return mc_model, encoder, decoder

def get_model_ensemble(n_mc=10):
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights.h5')
    decoder.load_weights('save/dec_weights.h5')

    models = []
    for name in filter(lambda x: 'mnist_cdrop_cnn' in x, os.listdir('save')):
        print('loading model {}'.format(name))
        model = load_drop_model('save/' + name)
        models.append(model)
    mc_model = U.MCEnsembleWrapper(models, n_mc=10)
    return mc_model, encoder, decoder

def get_ML_ensemble():
    
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights.h5')
    decoder.load_weights('save/dec_weights.h5')
    K.set_learning_phase(False)
    ms = []
    for name in filter(lambda x: 'mnist_cnn' in x, os.listdir('save')):
        print('loading model {}'.format(name))
        model = load_model('save/' + name)
        ms.append(model)

    model = U.MCEnsembleWrapper(ms, n_mc=1)
    return model, encoder, decoder 

    

def get_ML_models():
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights.h5')
    decoder.load_weights('save/dec_weights.h5')

    model = keras.models.load_model('save/mnist_cnn.h5')

    def get_results(X):
        preds = model.predict(X)
        ent = - np.sum(preds * np.log(preds + 1e-10), axis=-1)
        return preds, ent, np.zeros(ent.shape)

    model.get_results = get_results 
    return model, encoder, decoder


def get_ML_no_drop_models():
    _, encoder, decoder = define_VAE()
    encoder.load_weights('save/enc_weights.h5')
    decoder.load_weights('save/dec_weights.h5')

    model = keras.models.load_model('save/mnist_cnn_no_drop_run.h5')

    def get_results(X):
        preds = model.predict(X)
        ent = - np.sum(preds * np.log(preds + 1e-10), axis=-1)
        return preds, ent, np.zeros(ent.shape)

    model.get_results = get_results 
    return model, encoder, decoder


def make_interactive_plot(proj_x,
                          proj_y,
                          extent,
                          plot_bg,
                          decoder,
                          model,
                          title="",
                          bgcmap='gray',
                          bgalpha=0.9,
                          sccmap='Set3'):
    
    f, ax = plt.subplots(1,2)
    ax[0].scatter(proj_x[:,0],
                proj_x[:,1],
                c = proj_y.argmax(axis=1),
                marker=',',
                s=1,
                cmap=sccmap
    )

    ax[0].imshow(plot_bg,
                 cmap=bgcmap,
                 origin='lower',
                 alpha=bgalpha,
                 extent=extent,
    )
    ax[0].set_xlabel('First Latent Dimension')
    ax[0].set_ylabel('Second Latent Dimension')
    ax[0].set_title(title)

    latent_z1, latent_z2 = 0,0 #starting position
    proj = ax[1].imshow(decoder.predict(np.array([[latent_z1, latent_z2]])).squeeze(), cmap='gray_r')

    last_sample = None
    def on_click(click):
        global last_sample

        if click.xdata != None and click.ydata != None and click.inaxes==ax[0]:
            z1 = click.xdata
            z2 = click.ydata
            dream = decoder.predict(np.array([[z1, z2]]))
            pred,entropy,bald = model.get_results(dream)
            print("Predicted Class: {}, prob: {}".format(pred.argmax(axis=1), pred.max(axis=1)))
            print("Predictive Entropy: {}".format(entropy[0]))
            print("BALD Score:         {}".format(bald[0]))
            proj.set_data(dream.squeeze())
            plt.draw()
            last_sample = dream
    f.canvas.mpl_connect('button_press_event', on_click)

def make_plot(proj_x,
              proj_y,
              extent,
              plot_bg,
              decoder,
              model,
              title="",
              bgcmap='gray',
              bgalpha=0.9,
              sccmap='Set3'):
    
    f, ax = plt.subplots()
    ax.scatter(proj_x[:,0],
                proj_x[:,1],
                c = proj_y.argmax(axis=1),
                marker=',',
                s=1,
                cmap=sccmap,
               alpha=0.1
    )

    ax.imshow(plot_bg,
                 cmap=bgcmap,
                 origin='lower',
                 alpha=bgalpha,
                 extent=extent,
    )
    ax.set_xlabel('First Latent Dimension')
    ax.set_ylabel('Second Latent Dimension')
    ax.set_title(title)


if __name__ == '__main__':

    #model, encoder, decoder = get_models_3s_7s()
    
    #x_train, y_train, x_test, y_test = mnist_to_3s_and_7s(U.get_mnist())
    model, encoder, decoder = get_models()
    #model, encoder, decoder = get_model_ensemble(n_mc=20)
    
   # x_train, y_train, x_test, y_test = U.get_mnist()

   # model, encoder, decoder = get_ML_models()
    #model, encoder, decoder = get_ML_ensemble()
   # _, encoder, decoder = define_VAE()
   # encoder.load_weights('save/enc_weights.h5')
   # decoder.load_weights('save/dec_weights.h5')

   # model1 = load_drop_model('save/mnist_cdrop_cnn.h5')

   # model2 = keras.models.load_model('save/mnist_cnn.h5')
   # model, encoder, decoder = get_ML_no_drop_models()

    
    
    #proj_x_train = encoder.predict(x_train)
   # model = EnsembleWrapper([model1, model2])

    zmin, zmax = -10,10
    n_grid = 100
    preds, plot_ent, plot_bald = get_uncertainty_samples(model,
                                                         encoder,
                                                         decoder,
                                                         [zmin, zmax],
                                                         n_grid=n_grid)
    
    make_plot(proj_x_train,
              y_train,
              [zmin, zmax, zmin, zmax],
              plot_ent,
              decoder,
              model,
    )
    
    print('done')              
    #plt.savefig('overleaf-paper/figures/ML_dropout_uncertainty.pdf')
    plt.show()
