"""MultiProcess."""
import os
from multiprocessing import Process, Pool
import pandas as pd
import numpy as np
import time


def make_model(input_shape):
    """model."""
    import keras
    from keras.models import Sequential
    from keras.layers.core import Dense
    model = Sequential()
    model.add(Dense(7, input_shape=(input_shape, ), activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dense(7, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    sgd = keras.optimizers.SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def train(x, y):
    """train."""
    model = make_model((x.shape[1]))
    model.fit(np.array(x), np.array(y), epochs=100, verbose=0)
    return model.get_weights()

if __name__ == '__main__':
    print('load data...')
    cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    data = pd.read_csv('abalone.txt', index_col=False, names=cols)
    y = data['Rings']
    x = data.drop(['Rings', 'Sex'], axis=1)
    print('Start training...')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    t0 = time.time()
    with Pool(processes=10) as pool:
        multiple_res = [pool.apply_async(train, (x, y)) for i in range(10)]
        weights = [res.get() for res in multiple_res]
    # weights = []
    # for i in range(10):
    #    weights.append(train(x, y))

    print(time.time() - t0)
