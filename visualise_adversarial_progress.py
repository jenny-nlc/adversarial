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
from matplotlib import pyplot as plt
import src.mcmc as mcmc

#use the basic iterative method. This code is lifted from cleverhans, slightly modified in order
#to save the intermediate steps of the method.
class BasicIterativeMethod(Attack):
    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(BasicIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0
        
        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}
        steps = [x]
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x
            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(range(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm

            steps.append(x + eta) #don't see why this shouldn't work?

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x, steps #these are both tensors

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
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

def generate_plot(model_wrapper,data, x, nb_iter=10, eps=1, eps_iter=.1):
    """
    Generate the plot with data data for points x
    """
    
    bim = BasicIterativeMethod(model_wrapper, sess=K.get_session())
    input_tensor = K.placeholder(shape=(None,2))
    adv_end, adv_steps = bim.generate(input_tensor, eps = eps, nb_iter = nb_iter, eps_iter = eps_iter, ord = 2)

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


#test on a random-ish model for know
model, model_inputs = C.define_cdropout_model()
model.load_weights('server_output/toy_models/round_5/cdropout_toy_model_weights.h5')
data,labels = pickle.load(open('server_output/toy_models/round_5/toy_dataset.pickle', 'rb'))
hmc_model = C.define_standard_model()
hmc_weights = pickle.load(open('server_output/toy_models/round_5/hmc_ensemble_weights.pickle', 'rb'))
#define a closure for keras to use as a wrapper
def mc_keras_model(x):
    return K.mean(U.mc_dropout_preds(model, x, n_mc=C.N_MC), axis=0)
def get_bald(x):
    return U.BALD(U.mc_dropout_preds(model, x, n_mc=C.N_MC))

class HMCKerasModel:
    """
    The wrapper I wrote in the mcmc function returns a numpy array; if we
    want to differentiate the output of our model, we need to insantiate it
    in keras
    """
    def __init__(self):
        ensemble = []
        for ws in hmc_weights:
            model = C.define_standard_model()
            model.set_weights(ws)
            ensemble.append(model)
        self.ensemble = ensemble
    def __call__(self,x):
        return K.mean(K.stack([model(x) for model in self.ensemble]), axis=0) 

hmc_model = HMCKerasModel()
wrapper = CallableModelWrapper(hmc_model,'probs')
generate_plot(wrapper, data, data[:20], eps=.5, nb_iter=5,eps_iter=0.1) 
plt.show()
