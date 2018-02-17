import h5py
import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda, Reshape
from keras.models import Model
from scipy import stats

import src.utilities as U
from train_cdropout_3s_7s import mnist_to_3s_and_7s

BATCH_SIZE=128


def binary_logprob(y, y_):
    return y * np.log(y) + (1 - y) * np.log(1 - y_)

def define_CVAE(optim='adagrad', latent_dim=2):
    input_x = keras.layers.Input(shape=(28,28,1))
    input_c = keras.layers.Input(shape=(10,)) #one hot class distribution condition

    input_flatten = Flatten()(input_x)
    x = keras.layers.concatenate([input_flatten, input_c]) # input including the condition variable

    enc_h = Dense(512, activation='elu')(x)

    z_mu = Dense(latent_dim)(enc_h)
    z_logsigma = Dense(latent_dim)(enc_h)

    encoder = Model(inputs=[input_x, input_c], outputs=z_mu) #represent the latent space by the mean

    def sample_z(args):
        mu, logsigma = args
        return  0.5 * K.exp(logsigma / 2) * K.random_normal(shape=(K.shape(mu)[0], latent_dim)) + mu
    z = Lambda(sample_z, output_shape=(latent_dim,))([z_mu, z_logsigma]) 

    latent_input = keras.layers.Input(shape=(latent_dim,))
    dec_input = keras.layers.concatenate([latent_input, input_c])
    dec_h = Dense(512, activation='elu')(dec_input)
    dec_output = Dense(784, activation='sigmoid')(dec_h) 

    dec_reshaped = Reshape((28,28,1))(dec_output)
    decoder = Model(inputs=[latent_input, input_c],outputs=dec_reshaped)

    reconstruction = decoder([z, input_c])

    VAE = Model(inputs=[input_x, input_c], outputs=reconstruction)

    def vae_loss(inputs, reconstruction):
        x = K.batch_flatten(inputs)
        rec = K.batch_flatten(reconstruction)
        x_ent = K.sum(K.binary_crossentropy(x, rec), axis=-1)
        kl_div = 0.5 * K.sum(K.exp(z_logsigma) + K.square(z_mu) - z_logsigma - 1, axis=-1)
        return x_ent + kl_div
    
    VAE.compile(optimizer=optim, loss=vae_loss)

    return VAE, encoder, decoder

def generate(decoder, latent_dim, n_samples=10):
    z = np.random.randn(n_samples, latent_dim)
    classes = np.random.randint(0, high=10, size=n_samples)
    cond = keras.utils.to_categorical(classes, num_classes=10)

    samples = decoder.predict([z, cond])
    return samples

def make_grid(im_batch, rect):
    """
    Concatenate a batch of samples into an n by n image
    """
    h, w = rect
    ims = [im.squeeze() for im in im_batch]

    ims = [ims[i* w:(i+1)*w] for i in range(h)]

    ims = [np.concatenate(xs, axis=0) for xs in ims]

    ims = np.concatenate(ims, axis=1)
    return ims

def visualise_latent_space(decoder, n_grid=10, target= 2):

    grid = norm.ppf(np.linspace(0.01,0.99, n_grid))

    xx, yy = np.meshgrid(grid, grid)

    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
    y = keras.utils.to_categorical([target for _ in range(X.shape[0])], num_classes=10)
    Z = decoder.predict([X, y])

    Z = Z.reshape(n_grid, n_grid, 28,28)

    imgrid = np.concatenate(
        [np.concatenate([Z[i,j] for i in range(n_grid)], axis=1)
         for j in range(n_grid)], axis=0)
    plt.imshow(imgrid, cmap='gray_r')



def plot_examples(decoder, x_train, y_train):
    
    h = 20
    z_samps = np.random.randn(h * 10, latent_dim)
    classes = keras.utils.to_categorical(sum([[i] * h for i in range(10)], []), num_classes=10) 

    mnist_examples = np.concatenate([x_train[y_train.argmax(axis=1) == c][:h] for c in range(10)])
    samps = decoder.predict([z_samps,classes])
    plt.figure()
    plt.imshow(make_grid(samps, (10, h)), cmap='gray_r')
    plt.figure()
    plt.imshow(make_grid (mnist_examples , (10, h)), cmap= 'gray_r')
    plt.show()

def create_mmnist(decoder, n_train=60000, n_test = 30000):
    
    # generate the fake training data.
    
    mmnist_y_train = keras.utils.to_categorical(
        np.random.randint(0, high=10, size=n_train))   # c ~ Categorical[0..9]

    train_z  = np.random.randn(n_train, latent_dim)    # z ~ Normal([0 0 ..] ,eye(D))

    # This is only really the log probability *density* of the point; if we are being rigorous,
    # of course the probability of each point is zero. Imagine it's the probability in a ball of
    # radius machine epsilon or something.


    mmnist_x_train = decoder.predict([train_z, mmnist_y_train])
                       # prob of class * prob of latent variable * prob of image (in actual fact, this is the mean probability of the randomly binarised output)
    train_log_prob_d  = np.log(0.1) + stats.norm.logpdf(train_z).sum(axis=1) + binary_logprob(mmnist_x_train, mmnist_x_train).sum(axis=(1,2,3))

    mmnist_y_test = keras.utils.to_categorical(
        np.random.randint(0, high=10, size=n_test))    # c ~ Categorical[0..9]

    test_z  = np.random.randn(n_test, latent_dim)      # z ~ Normal([0 0 ..] ,eye(D))

    mmnist_x_test = decoder.predict([test_z, mmnist_y_test])

    test_log_prob_d  = np.log(0.1) + stats.norm.logpdf(test_z).sum(axis=1) + binary_logprob(mmnist_x_test, mmnist_x_test).sum(axis=(1,2,3))

    with h5py.File('datasets/manifold_mnist.h5', 'w') as f:
        f.create_dataset('x_train', data=mmnist_x_train)
        f.create_dataset('y_train', data=mmnist_y_train)
        f.create_dataset('train_logprob', data=train_log_prob_d)

        f.create_dataset('x_test', data=mmnist_x_test)
        f.create_dataset('y_test', data=mmnist_y_test)
        f.create_dataset('test_logprob', data=test_log_prob_d)



    


if __name__ == '__main__':
    latent_dim = 2
    x_train, y_train, x_test, y_test = U.get_mnist()

    VAE, encoder, decoder = define_CVAE(
        optim=keras.optimizers.Adam(),
        latent_dim=latent_dim
    )

    VAE.fit([x_train,y_train], x_train,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_data=([x_test, y_test], x_test))

    encoder.save_weights('save/mmnist_enc_weights_latent_dim_' + str(latent_dim) + '.h5')
    decoder.save_weights('save/mmnist_dec_weights_latent_dim_' + str(latent_dim) + '.h5')

       
