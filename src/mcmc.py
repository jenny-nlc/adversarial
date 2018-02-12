import keras
import numpy as np
from keras import backend as K


def reset_model(model: keras.models.Model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=K.get_session())
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=K.get_session())
    return

def HMC_ensemble_run(model: keras.models.Model,
            x_train: np.array,
            y_train: np.array,
            N_mc: int,
            ep: float,
            tau: int,
            burn_in: int,
            sample_every: int,
            return_extra=False,
            verbose=True,
            verbose_n = 100,
            ):
    """
    Takes a keras model and a dataset (x_train, y_train) and returns a list of numpy
    arrays to be used as weights for an ensemble predictor.
    """
    step_size = ep
    n_steps = tau

    Ws = model.weights
    lossfn = model.loss #this is kinda cheeky
    X = model.input
    Y_ = model.output
    Y = K.placeholder(shape=Y_.shape)
    L = K.sum(lossfn(Y, Y_)) #the sum accross the batch
    Gs = K.gradients(L, Ws)
    eval_loss_and_grads = K.function([X,Y], [L] + Gs)
    def get_loss_and_grads(x, y):
        res = eval_loss_and_grads([x, y])
        return res[0], res[1:]
    losses = []
    i = 0
    weights = []
    ensemble = []
    accept_n = 0
    ep_lo = 0.8 * step_size
    ep_hi = 1.2 * step_size

    tau_lo = int(0.5 * n_steps)
    tau_hi = int(1.5 * n_steps)
    
    obj = 0

    while len(ensemble) < N_mc:
        ep = ep_lo + (ep_hi - ep_lo) * np.random.random()
        tau = np.random.randint(tau_lo, high=tau_hi)

        obj, gs = get_loss_and_grads(x_train, y_train)
        losses.append(obj)


        if verbose and i % verbose_n == 0:
            acc = np.mean(model.predict(x_train).argmax(axis=1) == y_train.argmax(axis=1))
            accept_ratio = accept_n / i if i > 0 else 0
            print("iter: ", i, 'accuracy :', acc, 'loss: ', obj, 'accept_ratio: ', accept_ratio)

        i += 1 

        ps = [np.random.normal(size=w.shape) for w in Ws]  # momentum variables

        H = .5 * sum([np.sum(p ** 2) for p in ps]) + obj

        ws = [K.eval(w) for w in Ws]
        weights.append(ws)
        ws_old = [K.eval(w) for w in Ws]
        # store the values of the weights in case we need to go back
        for t in range(tau):
            for p, g in zip(ps, gs):
                p -= .5 * ep * g

            for w, p in zip(ws, ps):
                w += ep * p

            # evaluate new weights
            for (weight, value) in zip(Ws, ws):
                K.set_value(weight, value)
            obj, gs = get_loss_and_grads(x_train, y_train)

            for p, g in zip(ps, gs):
                p -= .5 * ep * g

        H_new = .5 * sum([np.sum(p ** 2) for p in ps]) + obj

        dH = H_new - H

        if (dH < 0) or np.random.rand() < np.exp(-dH):
            # in this case, we acccept the new values
            accept_n += 1
            pass
        else:
            # reverse the step
            for weight, value in zip(Ws, ws_old):
                K.set_value(weight, value)

        if i > burn_in and i % sample_every == 0:
           ensemble.append([K.eval(w) for w in Ws])

    if return_extra:
        weights = np.array(weights)
        return ensemble, losses, weights, accept_n / i
    else:
        return ensemble


def HMC_ensemble( model: keras.models.Model,
            x_train: np.array,
            y_train: np.array,
            N_mc=10,
            ep=5e-3,
            tau=1,
            burn_in=5000,
            sample_every=1000,
            samples_per_init=1,
            verbose=True,):
    """
    Trains an ensemble of models using HMC
    """
    ensemble = []
    i = 0
    while len(ensemble) < N_mc:
        if verbose: print("run: ", i)
        i += 1 
        still_todo = N_mc - len(ensemble) 
        n_samples = samples_per_init if still_todo > samples_per_init else still_todo
        reset_model(model)
        run_ensemble = HMC_ensemble_run(model,
                               x_train,
                               y_train,
                               n_samples,
                               ep,
                               tau,
                               burn_in,
                               sample_every,
                               verbose=verbose)
        ensemble += run_ensemble 
    return ensemble

def HMC_ensemble_predict(model, ensemble_weights, test_x):
    """
    Given a model and an ensemble of HMC weights, get the average prediction
    of the model
    """
    mc_preds = []
    for ews in ensemble_weights:
        weights = model.weights
        for w, ensemble_value in zip(weights, ews):
            K.set_value(w, ensemble_value)
        preds = model.predict(test_x)
        mc_preds.append(preds)
    return np.array(mc_preds)

def HMC(model: keras.models.Model,
        lossfn,
        x_train,
        y_train,
        x_test,
        N_models,
        epsilon,
        tau,
        burn_in,
        sample_every):
    ensemble_weights = HMC_ensemble(model, x_train, y_train, N_mc=N_models, ep=epsilon, tau=tau,burn_in=burn_in, sample_every=sample_every)
    preds = HMC_ensemble_predict(model,ensemble_weights, x_test)
    return preds


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from sklearn.datasets import make_classification
    data, labels = make_classification(n_features=2, n_redundant=0)
    labels = labels.reshape(-1, 1)
    data -= data.mean()
    data /= data.std()
    # test our implementation on a very simple logistic classifier.
    plt.scatter(data[:, 0], data[:, 1], c=labels.flatten())
    plt.show()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2,
                                 input_shape=(2,),
                                 activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2()))

    labels = keras.utils.to_categorical(labels, num_classes=2)
    xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    plot_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd')
    weights, losses, ws, accept_n = HMC_ensemble_run(model,data, labels, 10, 0.1, 150, 1000, 100, return_extra=True, verbose=True)
    mc_preds = HMC_ensemble_predict(model, weights, plot_x)
    plt.figure()
    plt.contourf(xx, yy, mc_preds.mean(axis=0).reshape(xx.shape))
    plt.scatter(data[:, 0], data[:, 1], c=labels.flatten())
    plt.show()

