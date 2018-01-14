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


H_ACT = 'elu'  # Might not be optimal, but a standard choice.
N_HIDDEN_UNITS = 500
N_DATA = 1000
LENGTH_SCALE = 5e-1 #setting a low length scale encourages uncertainty to be higher.
MODEL_PRECISION = 1 #classification problem: see Gal's Thesis
WEIGHT_DECAY = LENGTH_SCALE ** 2 / (2 * N_DATA * MODEL_PRECISION)
WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
DROPOUT_REGULARIZER = 1 / (MODEL_PRECISION * N_DATA)
N_MC = 50
N_CLASSES = 5

N_FEATURES = 50

ORD = np.inf
TRAIN_SPLIT = 0.8
N_INFORMATIVE=10
plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

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
if __name__ == "__main__":

    x,y = make_classification(n_classes = N_CLASSES,
                              n_samples = N_DATA,
                              n_features=N_FEATURES,
                              n_informative=N_INFORMATIVE,
                              class_sep = 1,
                              flip_y = 0.001)
    
    y = keras.utils.to_categorical(y)
    x = scale(x)
    split = int(TRAIN_SPLIT * N_DATA)
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
    model.fit(x_train,y_train,epochs=100, validation_data =(x_test,y_test))
    mc_model = U.MCModel(model, input_tensor, 50)

    fgr = FastGradientRange(CallableModelWrapper(mc_model,'probs'), sess=K.get_session())

    epsilons = np.linspace(0,1,50)
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
    print('done')

    fig, axes = plt.subplots(3,1)

    axes[0].plot(epsilons,adv_balds.mean(axis=1), c='b', label='Adversarial Direction')
    axes[0].plot(epsilons,rnd_balds.mean(axis=1), c='r', label='Random Direction')
    axes[0].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[0].set_ylabel('Average BALD')
    axes[0].legend()

    axes[1].plot(epsilons,adv_entropies.mean(axis=1), c='b', label='Adversarial Direction')
    axes[1].plot(epsilons,rnd_entropies.mean(axis=1), c='r', label='Random Direction')
    axes[1].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[1].set_ylabel('Average Entropy')
    axes[1].legend()

    #calculate accuracy
    adv_accs = np.equal(adv_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)
    rnd_accs = np.equal(rnd_preds.argmax(axis=-1), y_test.argmax(axis=1)[None,:]).astype(np.float).mean(axis=1)

    axes[2].plot(epsilons,adv_accs, c='b', label='Adversarial Direction')
    axes[2].plot(epsilons,rnd_accs, c='r', label='Random Direction')
    axes[2].set_xlabel('Step size ({} norm)'.format(ORD))
    axes[2].legend()
    axes[2].set_ylabel('Average Accuracy')

    plt.show()
