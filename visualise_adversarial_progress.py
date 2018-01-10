"""
The aim of this file is to create a plot that shows the trajectory of an adversarial example under standard
iterative generation methods.
"""

import pickle

import numpy as np

from keras import backend as K
import cdropout_vs_mcmc_train_models as C
import src.utilities as U
from cleverhans.attacks import Attack, FastGradientMethod
from cleverhans.model import CallableModelWrapper, Model
from cleverhans import utils_tf
import tensorflow as tf
from matplotlib import pyplot as plt
import src.mcmc as mcmc
import os



plt.rcParams['figure.figsize'] = 10,4
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

def fgm_range(x, preds, y=None, epsilons=[0.3], ord=np.inf,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    This is a slight modification of the fast gradient method to
    return a series of fgm attacks with a set of different epilons, in
    order to avoid the costly recomputation of the gradient.
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grads = [eps * normalized_grad for eps in epsilons]

    # Add perturbation to original example to obtain adversarial example
    adv_xs = [x + scaled_grad for scaled_grad in scaled_grads]

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    
    if (clip_min is not None) and (clip_max is not None):
        for adx_x in adv_xs:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_xs

class FastGradientRange(FastGradientMethod):
    """
    This is identical to the fast gradient attack in cleverhans,
    except that I override the generate method to use my modified
    attack above
    """
    def __init__(self, model, back='tf', sess=None):
        super(FastGradientRange, self).__init__(model, back, sess)
        self.feedable_kwargs = {'epsilons': np.array, #Iterable[np.float32]
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}

    def generate(self, x, **kwargs):
        """
        Replace the generate method of the FastGradientMethod class
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm_range(x, self.model.get_probs(x), y=labels, epsilons=self.epsilons,
                   ord=self.ord, clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None))
    
    def parse_params(self, epsilons=[0.3],  y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        # Save attack-specific parameters
        self.epsilons = epsilons
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

def generate_path_plots(model_wrapper,
                        model_entropy, # numpy array -> scalar
                        model_bald,    # numpy array -> scalar
                        data, x, epsilons):
    """
    Plot the bald and entropy of a single point becoming more and more adversarial
    """
    fgr = FastGradientRange(model_wrapper, sess=K.get_session())

    input_tensor = K.placeholder(shape=(None,2))
    adv_steps = fgr.generate(input_tensor, epsilons = epsilons, ord = 2)

    steps = [s.eval(session=K.get_session(), feed_dict={input_tensor: x}) for s in adv_steps]
    steps = np.array(steps).squeeze()

    entropies = model_entropy([steps])[0] 
    balds     = model_bald([steps])[0]
    f, axes = plt.subplots(1,3)

    axes[0].plot(entropies)
    axes[0].set_title('Predictive Entropy')
    axes[0].set_xlabel('Epsilon')
    axes[0].set_ylabel('Entropy')


    axes[1].plot(balds)
    axes[1].set_title('BALD Score')
    axes[1].set_xlabel('Epsilon')
    axes[1].set_ylabel('BALD')

    xx,yy = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))
    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
    preds = model_wrapper(input_tensor).eval(session=K.get_session(), feed_dict={input_tensor: X})
    #bald  = get_bald(input_tensor).eval(session=K.get_session(), feed_dict={input_tensor: X})
    #TODO: maybe add this capability later? 

    axes[2].imshow((preds[:,0].reshape(xx.shape) + 1e-6),
            cmap='gray',
            origin='lower',
            interpolation='bicubic',
            extent=[xx.min(),
                    xx.max(),
                    yy.min(),
                    yy.max()])
    axes[2].scatter(data[:,0], data[:,1], c=labels.argmax(axis=1))
    axes[2].plot(steps[:,0], steps[:,1], marker='+')
    return f


    
def generate_many_plot(model_wrapper,data, x, epsilons):
    """
    Generate the plot with data data for points x
    """
    fgr = FastGradientRange(model_wrapper, sess=K.get_session())

    input_tensor = K.placeholder(shape=(None,2))
    adv_steps = fgr.generate(input_tensor, epsilons = epsilons, ord = 2)

    steps = [s.eval(session=K.get_session(), feed_dict={input_tensor: x}) for s in adv_steps]
    steps = np.array(steps).squeeze()

    plt.figure()
    xx,yy = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))
    X = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
    preds = model_wrapper(input_tensor).eval(session=K.get_session(), feed_dict={input_tensor: X})
    #bald  = get_bald(input_tensor).eval(session=K.get_session(), feed_dict={input_tensor: X})
    #TODO: maybe add this capability later? 

    plt.imshow((preds[:,0].reshape(xx.shape) + 1e-6),
            cmap='gray',
            origin='lower',
            interpolation='bicubic',
            extent=[xx.min(),
                    xx.max(),
                    yy.min(),
                    yy.max()])
    plt.scatter(data[:,0], data[:,1], c=labels.argmax(axis=1))
    for i in range(len(x)):
        plt.plot(steps[:,i,0], steps[:,i,1], marker='+')

class HMCKerasModel:
    """
    The wrapper I wrote in the mcmc function returns a numpy array; if we
    want to differentiate the output of our model, we need to insantiate it
    in keras
    """
    def __init__(self, weights):
        ensemble = []
        for ws in hmc_weights:
            model = C.define_standard_model()
            model.set_weights(ws)
            ensemble.append(model)
        self.ensemble = ensemble
    def __call__(self,x):
        return K.mean(K.stack([model(x) for model in self.ensemble]), axis=0) 
    def generate_closures(self):
        input_tensor = K.placeholder(shape=(None,2))
        mc_preds = K.stack([model(input_tensor) for model in self.ensemble])
        pred_H = U.predictive_entropy(mc_preds)
        exp_H  = U.expected_entropy(mc_preds)

        get_entropy = K.function([input_tensor], [pred_H])
        get_bald    = K.function([input_tensor], [pred_H - exp_H])

        return get_entropy, get_bald

LOAD_PATH='server_output/toy_models/round_6'
#test on a random-ish model for know
model, model_inputs = C.define_cdropout_model()
model.load_weights(os.path.join(LOAD_PATH,'cdropout_toy_model_weights.h5'))
data,labels = pickle.load(open(os.path.join(LOAD_PATH,'toy_dataset.pickle'), 'rb'))
#define a closure for keras to use as a wrapper
def mc_keras_model(x):
    return K.mean(U.mc_dropout_preds(model, x, n_mc=C.N_MC), axis=0)
def get_closures():
    input_tensor = K.placeholder(shape=(None,2))
    mc_preds = U.mc_dropout_preds(model, input_tensor, n_mc=C.N_MC)
    pred_H = U.predictive_entropy(mc_preds)
    exp_H  = U.expected_entropy(mc_preds)

    get_entropy = K.function([input_tensor], [pred_H])
    get_bald    = K.function([input_tensor], [pred_H - exp_H])

    return get_entropy, get_bald

epsilons = np.linspace(0,1.5,10)
e, b = get_closures()
f1 = generate_path_plots(CallableModelWrapper(mc_keras_model, 'probs'),e,b, data, data[5:6], epsilons = epsilons)

hmc_weights = pickle.load(open(os.path.join(LOAD_PATH, 'hmc_ensemble_weights.pickle'), 'rb'))

       

hmc_model = HMCKerasModel(hmc_weights)
wrapper = CallableModelWrapper(hmc_model,'probs')
e,b = hmc_model.generate_closures()
f2 = generate_path_plots(wrapper, e,b,data, data[5:6], epsilons=epsilons) 
 o
