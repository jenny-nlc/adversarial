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


def HMC_run(model: keras.models.Model,
            lossfn,  # Tensor -> Tensor -> Tensor
            x_train: np.array,
            y_train: np.array,
            x_test: np.array,
            n_runs: int,
            ep: float,
            tau: int,
            burn_in: int,
            between_sample: int,
            return_extra=False,
            verbose=True,
            ):
    """This takes a keras model and uses HMC to do proper bayesian inference
    over the inputs x_test.
    Caveat Emptor- this takes a *very long time*
    """
    # set up input and output tensors
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

    mc_preds = np.zeros([x_test.shape[0]] + list(Y.shape[1:]))
    n_mc = 0
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
        weights.append([ws[0][0], ws[0][1], ws[1]])
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

        if i > 1000:
            # burn in
            if i % 500 == 0:
                mc_preds += model.predict(x_test)
                n_mc += 1

    mc_preds /= n_mc
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
    mc_preds = None

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
        mc_preds = run_mc_preds if mc_preds is None else mc_preds + run_mc_preds
    mc_preds /= n_runs
    return mc_preds


if __name__ == "__main__":

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
    mc_preds = HMC(model,
                   keras.losses.binary_crossentropy,
                   data,
                   labels,
                   plot_x,
                   10,
                   15000,
                   2,
                   1,
                   10000,
                   1000)

    plt.figure()
    plt.contourf(xx, yy, mc_preds.reshape(xx.shape))
    plt.scatter(data[:, 0], data[:, 1], c=labels.flatten())
    plt.show()
