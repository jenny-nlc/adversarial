
import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda, Reshape
from keras.models import Model

import src.utilities as U
from train_cdropout_3s_7s import mnist_to_3s_and_7s

BATCH_SIZE=128


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
        x = K.flatten(inputs)
        rec = K.flatten(reconstruction)
        x_ent = keras.metrics.binary_crossentropy(x, rec)
        kl_div = 0.5 * K.sum(K.exp(z_logsigma) + K.square(z_mu) - z_logsigma - 1, axis=-1)
        return 28 * 28 * x_ent + kl_div
    
    VAE.compile(optimizer=optim, loss=vae_loss)

    return VAE, encoder, decoder

if __name__ == '__main__':
    latent_dim = 64
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

       
