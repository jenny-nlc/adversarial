"""
This file is to test whether the theory of https://arxiv.org/pdf/1608.07690.pdf
is a reasonable explanation of BALD as an adversarial attack detector.
"""
import keras
import keras.backend as K
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale
import src.utilities as U
from keras.layers import Dense
from src.concrete_dropout import ConcreteDropout
from visualise_adversarial_progress import FastGradientRange
from matplotlib import pyplot as plt
from cleverhans.model import CallableModelWrapper



H_ACT = 'relu'  # Might not be optimal, but a standard choice.
N_HIDDEN_UNITS = 500
N_DATA = 100
LENGTH_SCALE = 1e-2 #setting a low length scale encourages uncertainty to be higher.
MODEL_PRECISION = 1 #classification problem: see Gal's Thesis
WEIGHT_DECAY = LENGTH_SCALE ** 2 / (2 * N_DATA * MODEL_PRECISION)
WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
DROPOUT_REGULARIZER = 1 / (MODEL_PRECISION * N_DATA)
N_MC = 50
N_CLASSES = 2  # number of classes

N_FEATURES = 100

ORD = 2
N_SAMPLES = 1500
TRAIN_SPLIT = 0.9
N_INFORMATIVE=10
def define_cdropout_model():
    """
    Define the cdropout model. This is written as a function for easier loading
    from the plotting script
    """
    inputs = keras.layers.Input(shape=(N_FEATURES,))

    h1 = Dense(N_HIDDEN_UNITS, activation=H_ACT)(inputs)
    h2 = ConcreteDropout(Dense(N_HIDDEN_UNITS, activation=H_ACT),
                         weight_regularizer=WEIGHT_REGULARIZER,
                         dropout_regularizer=DROPOUT_REGULARIZER)(h1)
    predictions = ConcreteDropout(Dense(N_CLASSES, activation='softmax'),
                                  weight_regularizer=WEIGHT_REGULARIZER,
                                  dropout_regularizer=DROPOUT_REGULARIZER)(h2)
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    return model, inputs

class MCModel:
    def __init__(self,model, input_tensor, n_mc):
        self.model = model
        self.input = input_tensor
        self.n_mc=n_mc
        self.mc_preds_t = U.mc_dropout_preds(self.model, self.input, n_mc=n_mc)
        self.predictive_entropy_t = U.predictive_entropy(self.mc_preds_t)
        self.expected_entropy_t   = U.expected_entropy(self.mc_preds_t)
        self.bald_t = self.predictive_entropy_t - self.expected_entropy_t

    def get_results(self,x):
        f = K.function([self.input],
                       [K.mean(self.mc_preds_t, axis=0),
                        self.predictive_entropy_t, self.bald_t])
        return f([x])
    def predict(self, x):
        return self.get_results(x)[0]
    def __call__(self, x):
        return K.mean(U.mc_dropout_preds(self.model, x, n_mc = self.n_mc), axis=0)
if __name__ == "__main__":

    x,y = make_classification(n_classes = N_CLASSES,
                              n_samples = N_SAMPLES,
                              n_features=N_FEATURES,
                              n_informative=N_INFORMATIVE)
    
    y = keras.utils.to_categorical(y)
    x = scale(x)
    split = int(TRAIN_SPLIT * N_SAMPLES)
    x_train = x[:split]
    y_train = y[:split]
    x_test  = x[split:]
    y_test  = y[split:]
    
    model, input_tensor = define_cdropout_model()

    model.compile(
        optimizer=keras.optimizers.SGD(
            lr = 0.05,
            momentum=0.1,
            decay=0.0001,
            nesterov=True,
            ),
        loss = keras.losses.categorical_crossentropy,
        metrics=['accuracy']
        )
    model.fit(x_train,y_train,epochs=150, validation_data =(x_test,y_test))
    mc_model = MCModel(model, input_tensor, 50)

    fgr = FastGradientRange(CallableModelWrapper(mc_model,'probs'), sess=K.get_session())

    epsilons = np.linspace(0,1,20)
    adv_steps = fgr.generate(input_tensor, epsilons=epsilons, ord=ORD) 

    advs = [s.eval(session=K.get_session(), feed_dict={input_tensor: x_test}) for s in adv_steps]

    adv_preds = []
    adv_entropies = []
    adv_balds = []
    for a in advs:
        p,e,b = mc_model.get_results(a) 
        adv_preds.append(p)
        adv_entropies.append(e)
        adv_balds.append(b)
    adv_preds = np.array(adv_preds)
    adv_entropies = np.array(adv_entropies)
    adv_balds = np.array(adv_balds)

    #observe the same for a random pertubation
    eta = np.random.normal(size=x_test.shape)
    #normalise the pertubation
    eta /= np.linalg.norm(eta, axis=1, ord=ORD)[:, None]

    #apply the pertubations at various step sizes
    perturbs = [x_test + ep * eta for ep in epsilons]
    rnd_preds = []
    rnd_entropies = []
    rnd_balds = [] 

    for a in perturbs:
        p,e,b = mc_model.get_results(a) 
        rnd_preds.append(p)
        rnd_entropies.append(e)
        rnd_balds.append(b)
    rnd_preds = np.array(rnd_preds)
    rnd_entropies = np.array(rnd_entropies)
    rnd_balds = np.array(rnd_balds)

    plt.figure()
    plt.plot(epsilons,adv_balds.mean(axis=1), c='b', label='Adversarial Direction')
    plt.plot(epsilons,rnd_balds.mean(axis=1), c='r', label='Random Direction')
    plt.title('BALD / epsilon')
    plt.xlabel('Step size ({} norm)'.format(ORD))
    plt.ylabel('Average BALD')
    plt.legend()

    plt.figure()
    plt.plot(epsilons,adv_entropies.mean(axis=1), c='b', label='Adversarial Direction')
    plt.plot(epsilons,rnd_entropies.mean(axis=1), c='r', label='Random Direction')
    plt.title('Entropy / epsilon')
    plt.xlabel('Step size ({} norm)'.format(ORD))
    plt.ylabel('Average Entropy')
    plt.legend()

    #calculate accuracy
    adv_accs = np.equal(adv_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)
    rnd_accs = np.equal(rnd_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)

    plt.figure()
    plt.plot(epsilons,adv_accs, c='b', label='Adversarial Direction')
    plt.plot(epsilons,rnd_accs, c='r', label='Random Direction')
    plt.title('Accuracy/ epsilon')
    plt.xlabel('Step size ({} norm)'.format(ORD))
    plt.legend()
    plt.ylabel('Average Accuracy')
    plt.show()
