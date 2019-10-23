from losses import keras_quantile_loss
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


class KerasQuantiles:

    def __init__(self,
                 x_shape=(1,),
                 y_shape=(1,),
                 shared_units=(4, 4),
                 quantile_units=(8,),
                 activation="relu",
                 quantiles=None,
                 optimizer=Adam,
                 lr=.01,
                 early_stop=EarlyStopping(monitor='val_loss', patience=20),
                 batch_size=64,
                 epochs=20):

        """
        :param x_shape: The shape of the input
        :param y_shape: The shape of the output
        :param shared_units: the number of shared layers in the NN
        :param quantile_units: the number of layers dedicated to quantiles
        :param activation: the activation function, default is ReLu
        :param quantiles: the quantile(s)
        :param optimizer: optimizer for the whole model, default is Adam
        :param lr: learning rate
        :param early_stop: early stop when the loss doesn't improve
        :param batch_size: batch size, number of examples fed at once to the NN
        :param epochs: number of epochs to train for
        """

        if quantiles is None:
            quantiles = [0.5]
        self._model_instance = None
        self.in_dim = x_shape
        self.out_dim = y_shape
        self.shared_units = shared_units
        self.quantile_units = quantile_units
        self.activation = activation
        self.optim = optimizer
        self.quantiles = quantiles
        self.lr = lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.epochs = epochs

    def _model(self):

        """
        Creation of the model using Keras Functional API. This allows to loop over layers for creation
        and to define intermediate outputs for shared layers and quantile layers. The use is described
        in the Keras functional API guide.
        :return: the compiled model with quantile loss
        """

        inputs = Input(shape=(self.in_dim,), name="Input")

        intermediate = inputs

        for idx, units in enumerate(self.shared_units):
            intermediate = Dense(
                units=units, activation=self.activation, name=f'dense_{idx}'
            )(inputs)

        outputs = [intermediate for _ in self.quantiles]

        for idx, units in enumerate(self.quantile_units):
            outputs = [
                Dense(units, activation=self.activation, name=f'q_{q}_dense_{idx}')(output)
                for q, output in zip(self.quantiles, outputs)
            ]
        outputs = [
            Dense(self.out_dim, name=f'q_{q}_out')(output) for q, output in zip(self.quantiles, outputs)
        ]
        model = Model(inputs, outputs, name='Quantile Regressor')

        model.compile(
            optimizer=self.optim(lr=self.lr), loss=[keras_quantile_loss(q) for q in self.quantiles]
        )

        return model

    def _init_model(self):
        """
        instantiation of the model, mostly hidden to the end user but ensures proper behavior,
        it calls the _model method to create the model.
        :return: instantiated model
        """
        self._model_instance = self._model()
        return self._model_instance

    @property
    def model(self):
        return self._model_instance or self._init_model()

    def fit(self, X, y, **kwargs):
        """

        :param X: input to train on, often called X_train
        :param y: target variable, often called y_train
        :param kwargs: kwargs for Keras fit method
        :return: fitted model
        """
        self._init_model()
        y = [y for _ in self.quantiles]
        fit_kwargs = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        fit_kwargs.update(kwargs)

        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X, **kwargs):
        """

        :param X: input to make prdictions on, often called X_test
        :param kwargs: kwargs for Keras predict method
        :return: predictions
        """
        predict_kwargs = dict(batch_size=self.batch_size, )
        predict_kwargs.update(kwargs)

        return np.hstack(self.model.predict(X, **predict_kwargs)).reshape(X.shape[0], -1)
