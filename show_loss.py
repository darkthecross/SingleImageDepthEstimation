import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from dbReader_simplified import dbReader

def main(s):
    if len(s) == 0:
        print('Error: please specify model.')
    else:
        f = open('analyst_new/loss_' + s[1] + '.pkl', 'rb')
        [ltr, lte] = pickle.load(f)
        tr, = plt.plot(ltr[3:],label='training error')
        te, = plt.plot(lte[3:],label='test error')
        plt.legend(handles=[tr, te])
        plt.show()

if __name__ == '__main__':
    main(sys.argv)
