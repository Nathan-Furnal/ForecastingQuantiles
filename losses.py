import keras.backend as K
import functools


def _tilted_loss_tensor(q, y_true, y_pred):
    """
    :param q: quantile, a float in (0,1)
    :param y_true: the true value of the array
    :param y_pred: the predicted value
    :return: a scalar, the maximum error
    """
    err = (y_pred - y_true)
    return K.maximum(-q * err, (1 - q) * err)


def _tilted_loss_scalar(q, y_true, y_pred):
    """
    :param q: quantile, a flaot in (0,1)
    :param y_true: the true value of the array
    :param y_pred: the predicted value
    :return: the mean of the tensor over many tilted losses, a scalar
    """
    return K.mean(_tilted_loss_tensor(q, y_true, y_pred), axis=-1)


def keras_quantile_loss(q):
    """
    :param q: a quantile in (0,1)
    :return: the precedent function computed for a given quantile
    the use of functools.partial allows to pass only the quantile as argument
    """
    func = functools.partial(_tilted_loss_scalar, q)
    func.__name__ = f'quantile loss, q={q}'
    return func
