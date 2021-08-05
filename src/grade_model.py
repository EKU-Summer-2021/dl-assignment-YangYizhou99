'''
library
'''
import configparser
import os
from tensorflow import keras


class GradeModel:
    '''
        this class is model
    '''
    def __init__(self):
        '''
            this method initialize
        '''
        config = configparser.ConfigParser()
        config.read(os.path.\
                    join(os.path.\
                         dirname(__file__), '../config', 'grade.conf'))
        neurons=config['Parameters']['neurons']
        self.model=keras.models.Sequential([
            keras.layers.Flatten(input_shape=[56,]),
            keras.layers.Dense(neurons, activation="relu"),
            keras.layers.Dense(neurons, activation="relu"),
            keras.layers.Dense(21, activation="softmax")
                ])
        self.model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])
        self.nothing=0

    def do_right2(self):
        '''
            this method do nothing
        '''
        return self.nothing

    def do_left2(self):
        '''
            this method do nothing
        '''
        return self.nothing
