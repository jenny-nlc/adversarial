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
            ):
    """
    Takes a keras model and a dataset (x_train, y_train) and returns a list of numpy
    arrays to be used as weights for an ensemble predictor.
    """
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
    while len(ensemble) < N_mc:
        if verbose and i % 1000 == 0:
            print("iter: ", i)
        i += 1 
        obj, gs = get_loss_and_grads(x_train, y_train)
        losses.append(obj)
        ps = [np.random.normal(size=w.shape) for w in Ws]  # momentum variables

        H = .5 * sum([np.sum(p ** 2) for p in ps]) + obj

        ws = [K.eval(w) for w in Ws]
        weights.append(ws)
        ws_old = [K.eval(w) for w in Ws]
        # store the values of the weights in case we need to go back
        for t in range(tau):
            for p, g in zip(ps, gs):
                p - .5 * ep * g

            for w, p in zip(ws, ps):
                w += ep * p

            # evaluate new weights
            for (weight, value) in zip(Ws, ws):
                K.set_value(weight, value)
            obj, gs = get_loss_and_grads(x_train, y_train)

            for p, g in zip(ps, gs):
                p - .5 * ep * g

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
            ep=5-3,
            tau=1,
            burn_in=5000,
            sample_every=1000,
            samples_per_init=1,
            verbose=True,):
    ensemble = []
    i = 0
    while len(ensemble) < N_mc:
        if verbose: print("run: ", i)
        i += 1 
        reset_model(model)
        run_ensemble = HMC_ensemble_run(model,
                               x_train,
                               y_train,
                               samples_per_init,
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
    cma = None
    n = 1
    for ews in ensemble_weights:
        weights = model.weights
        for w, ensemble_value in zip(weights, ews):
            K.set_value(w, ensemble_value)
        preds = model.predict(test_x)
        if cma is None:
            cma = preds
        else:
            n += 1
            cma += (preds - cma) / n
    return cma



def HMC_run(model: keras.models.Model,
            lossfn,  # Tensor -> Tensor -> Tensor
            x_train: np.array,
            y_train: np.array,
            x_test: np.array,
            n_runs: int,
            ep: float,
            tau: int,
            burn_in: int,
            sample_every: int,
            return_extra=False,
            verbose=True,
            ):
    """This takes a keras model and uses HMC to do proper bayesian inference
    over the inputs x_test.
    Caveat Emptor- this takes a *very long time*
    """
    # set up input and output tensors
    # N.B: this model.input/model.output trick, as well as passing in the lossfn,
    # is a bit of a hack and only really works on Sequential models. If you want
    # to adapt this code to be more generic, it might be better to pass in a closure
    # taking a model and the data and returning the loss and it's gradients, since that
    # is all the method actually needs, along with the weights.
    X = model.input
    Y_ = model.output
    Y = K.placeholder(shape=Y_.shape)  # to hold the true y_train
    L = K.sum(lossfn(Y, Y_))
    Ws = model.weights  # :: [K.Tensor]
    Gs = K.gradients(L, Ws)  # :: [K.Tensor]
    eval_grads = K.function([X, Y], [L] + Gs)

    def get_grads(x, y):
        res = eval_grads([x, y])
        return res[0], res[1:]
    losses = []

    mc_preds = []
    weights = []
    accept_n = 0

    for i in range(n_runs):
        if verbose and i % 1000 == 0:
            print("iter: ", i)
        obj, gs = get_grads(x_train, y_train)
        losses.append(obj)
        ps = [np.random.normal(size=w.shape) for w in Ws]  # momentum variables

        H = .5 * sum([np.sum(p ** 2) for p in ps]) + obj

        ws = [K.eval(w) for w in Ws]
        weights.append(ws)
        ws_old = [K.eval(w) for w in Ws]
        # store the values of the weights in case we need to go back
        for t in range(tau):
            for p, g in zip(ps, gs):
                p - .5 * ep * g

            for w, p in zip(ws, ps):
                w += ep * p

            # evaluate new weights
            for (weight, value) in zip(Ws, ws):
                K.set_value(weight, value)
            obj, gs = get_grads(x_train, y_train)

            for p, g in zip(ps, gs):
                p - .5 * ep * g

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

        if i >= burn_in:
            # burn in
            if i % sample_every == 0:
                mc_preds.append(model.predict(x_test))

    if return_extra:
        weights = np.array(weights)
        return mc_preds, losses, weights, accept_n / n_runs
    else:
        return mc_preds


def HMC(model: keras.models.Model,
        lossfn,
        x_train,
        y_train,
        x_test,
        n_runs,
        run_length,
        epsilon,
        tau,
        burn_in,
        sample_every):
    mc_preds = []

    for i in range(n_runs):
        print("run: ", i)
        reset_model(model)
        run_mc_preds = HMC_run(model,
                               lossfn,
                               x_train,
                               y_train,
                               x_test,
                               run_length,
                               epsilon,
                               tau,
                               burn_in,
                               sample_every)
        mc_preds += run_mc_preds 
    return np.array(mc_preds) #n_mc x batch_size x n_classes


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
    model.add(keras.layers.Dense(1,
                                 input_shape=(2,),
                                 activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2()))
    xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    plot_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

    mc_preds = HMC(model,
                   keras.losses.binary_crossentropy,
                   data,
                   labels,
                   plot_x,
                   2,
                   10001,
                   2,
                   1,
                   10000,
                   1)

    plt.figure()
    plt.contourf(xx, yy, mc_preds.mean(axis=0).reshape(xx.shape))
    plt.scatter(data[:, 0], data[:, 1], c=labels.flatten())
    plt.show()

