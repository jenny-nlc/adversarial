"""
Preform a similar function to the compare_adv_to_random script, but on mnist.
"""
import keras
import keras.backend as K
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale
import src.utilities as U
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from src.concrete_dropout import ConcreteDropout
from visualise_adversarial_progress import FastGradientRange
from cleverhans.model import CallableModelWrapper
import pickle
import os

def load_model():
    model = Sequential()

    act_fn = 'relu'
    input_shape = (28,28,1)
    num_classes = 10


    model.add(ConcreteDropout(Conv2D(32, kernel_size=(3,3),
        activation=act_fn),
        input_shape=input_shape))
    model.add(ConcreteDropout(Conv2D(64, (3,3), activation=act_fn)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(ConcreteDropout(Dense(128, activation=act_fn)))
    model.add(ConcreteDropout(Dense(num_classes, activation='softmax')))

    model.load_weights('mnist_cdrop_cnn.h5')
    return model

def eval_perturbations(perturbations, mc_model, batch_size=256): 
    preds = []
    entropies = []
    balds = []
    for a in perturbations:
        zipres = [mc_model.get_results(x) for x in batch_gen(a, batch_size=batch_size)] 
        bpred = np.concatenate([x[0] for x in zipres]) 
        bentropy = np.concatenate([x[1] for x in zipres])
        bbald = np.concatenate([x[2] for x in zipres])
        preds.append(bpred)
        entropies.append(bentropy)
        balds.append(bbald)
    preds = np.array(preds)
    entropies = np.array(entropies)
    balds = np.array(balds)
    return preds, entropies, balds

def batch_gen(array, batch_size=256):
    N = array.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    bs=batch_size
    return (array[i*bs:(i+1)*bs] for i in range(n_batches))

if __name__ == '__main__':
    batch_size = 100
    ORD = 2
    epsilons = np.linspace(0,1,40)


    x_train, y_train, x_test, y_test = U.get_mnist()
    x_test = x_test[:1000]     
    model = load_model()
    input_tensor = model.input
    mc_model = U.MCModel(model, input_tensor, n_mc = 50 )

    fgr = FastGradientRange(CallableModelWrapper(mc_model, 'probs'), sess=K.get_session())


    adv_steps = fgr.generate(input_tensor, epsilons=epsilons, ord=ORD)

    #create adversaral examples
    advs = [np.concatenate([s.eval(session=K.get_session(), feed_dict={input_tensor: x})
                            for x in batch_gen(x_test, batch_size=batch_size)])
            for s in adv_steps]
    adv_preds, adv_entropies, adv_balds = eval_perturbations(advs, mc_model, batch_size=batch_size) 
     
    

    #observe the same for a random pertubation of unit norm and step size epsilon 
    eta = np.random.normal(size=x_test.shape)
    #normalise the pertubation
    eta = eta.reshape(x_test.shape[0], -1)
    eta /= np.linalg.norm(eta, axis=1, ord=ORD)[:, None]
    eta = eta.reshape(x_test.shape)
    #apply the pertubations at various step sizes

    perturbs = [x_test + ep * eta for ep in epsilons]

    rnd_preds, rnd_entropies, rnd_balds = eval_perturbations(perturbs, mc_model, batch_size=batch_size)

    save_path = U.create_unique_folder(os.path.join('save', 'mnist_pertubations'))

    with open(os.path.join(save_path, 'mnist_pertub_data.pickle'),'wb') as f:
        pickle.dump([epsilons,
                     adv_preds,
                     adv_entropies,
                     adv_balds,
                     rnd_preds,
                     rnd_entropies,
                     rnd_balds], f)
