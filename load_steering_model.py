#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Conv2D
from keras import backend as K

def get_model(time_len=1):
  """
  This needs to be updated to keras 2 for me to be able to load it...
  """
  K.set_image_data_format('channels_first')
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
  model.add(ELU())
  model.add(Conv2D(32, (5, 5),strides=(2, 2), padding="same"))
  model.add(ELU())
  model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model

