import pandas as pd
import os
import numpy as np

from keras import losses
from keras.layers import Input, Embedding, Dense, Dropout, Flatten
from keras.models import Model
from keras.engine import Layer
import keras.backend as K

import tensorflow as tf


def reverse_gradient(X, hp_lambda):
    """Flips the sign of the gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    """Flips the sign of the gradient during training. Wraps the
    reverse_gradient function to a Keras layer."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"hp_lambda": self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GradientReversalModel:
    """
    Model with a few hidden layers and two outputs: in this case predefined for COMPAS.
    """

    def load_trained_model(self, path: str, hp_lambda: int):
        """
        """
        self.define_model(hp_lambda)
        self.model.load_weights(path)

    def get_model(self):
        """
        return the underlying model
        """
        return self.model

    def predict(self, X):
        """
        Wrapper around keras' predict function.
        """
        self.model.predict(X)

        return self.model.predict(X)

    def define_model(self, hp_lambda: int):
        """
        Creates a model in with a lambda value that's provided by the caller.
        """

        input = Input(batch_shape=(None, 11), name="input")

        x = Dense(64, activation="relu")(input)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        output = Dense(1, activation="linear", name="output")(x)
        if hp_lambda > 0:
            # Full model with branch for protected attribute
            Flip = GradientReversal(hp_lambda=hp_lambda)
            dann_in = Flip(x)
            dann_out = Dense(2)(dann_in)

            output2 = Dense(2, activation="softmax", name="output2")(dann_out)

            self.model = Model(inputs=[input], outputs=[output, output2])

            self.model.compile(
                optimizer="adam",
                loss={"output": losses.MSE, "output2": losses.binary_crossentropy},
                metrics={"output": losses.MSE, "output2": "acc"},
            )
        else:
            # Model without branch for protected attribute
            self.model = Model(inputs=[input], outputs=[output])

            self.model.compile(
                optimizer="adam",
                loss={"output": losses.MSE},
                metrics={"output": losses.MSE},
            )

