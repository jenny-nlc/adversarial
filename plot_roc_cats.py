import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
import src.utilities as U

plt.rcParams['figure.figsize'] = 8, 8
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='file to load and plot')
    parser.add_argument('save', action='store_true', help='whether to save the plots to disk')

    args = parser.parse_args()
    
    db = h5py.File(args.filename, 'r')

    #deal with the example images
    ims = db['example_imgs'].value
    ims = U.imagenet_deprocess(ims) 
    #these are a little bit confusing; only show some images

    ims = ims[:224 * 5, :, :]
    #basic plot, showing the generated adversarial images
    plt.figure(1)
    plt.imshow(ims)
    plt.axis('off')

    #
    plt.figure(2)
    imsdiff = ims - np.concatenate([ims[:, :224, :] for _ in range(3)], axis=1)
    imsdiff[:, :224, :] = ims[:, :224, :]
    plt.imshow(imsdiff)
    plt.axis('off')


    plt.figure(3)
    # plot the ROC curves
    plt.plot(db['Deterministic Model']['entropy_fpr'],
             db['Deterministic Model']['entropy_tpr'],
             label='Entropy (deterministic model)')
    
    plt.plot(db['MC Model']['entropy_fpr'],
             db['MC Model']['entropy_tpr'],
             label='Entropy (MC model)')

    plt.plot(db['MC Model']['bald_fpr'],
             db['MC Model']['bald_tpr'],
             label='Mutual Information (MC model)')
 
    plt.plot(db['Deterministic Model']['entropy_fpr_succ'],
             db['Deterministic Model']['entropy_tpr_succ'],
             label='Entropy (deterministic model) (succ) ')
    
    plt.plot(db['MC Model']['entropy_fpr_succ'],
             db['MC Model']['entropy_tpr_succ'],
             label='Entropy (MC model) (succ) ')

    plt.plot(db['MC Model']['bald_fpr_succ'],
             db['MC Model']['bald_tpr_succ'],
             label='Mutual Information (MC model) (succ) ')
    plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), color='k', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
#    plt.savefig('overleaf-paper/figures/cats_dogs_roc.pdf')

    plt.figure(4)
    # plot the PR curves
    plt.plot(db['Deterministic Model']['entropy_rec'],
             db['Deterministic Model']['entropy_prec'],
             label='Entropy (deterministic model)')
    
    plt.plot(db['MC Model']['entropy_rec'],
             db['MC Model']['entropy_prec'],
             label='Entropy (MC model)')

    plt.plot(db['MC Model']['bald_rec'],
             db['MC Model']['bald_prec'],
             label='Mutual Information (MC model)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    print('Accuracy det model: ',db['Deterministic Model']['adv_accuracy'].value)
    print('Accuracy mc model: ',db['MC Model']['adv_accuracy'].value)
    plt.show()

    import pdb; pdb.set_trace()
    db.close()
