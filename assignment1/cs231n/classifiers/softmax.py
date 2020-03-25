from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W

    Функция потери Softmax, наивная реализация (с петлями)

    Входные сигналы имеют размер D, есть классы C, и мы работаем на мини-пакетах
    из N примеров.

    Входные:
    - W: массив numpy формы (D, C), содержащий веса.
    - X: массив numpy формы (N, D), содержащий минибатч данных.
    - y: массив numpy формы (N,), содержащий обучающие метки; y[i] = c означает что X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) сила регуляризации

    Возвращает кортеж из:
    - потеря как float
    - градиент по отношению к весам W; массив той же формы, как и W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # http://cs231n.github.io/linear-classify/#softmax

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        f = X[i].dot(W) # матрица вероятностей
        f -= np.max(f)# нормализация

        # Промежуточные вычисления внутри формулы софтмакс
        F_y_i = f[y[i]]
        znamenatel = np.sum(np.exp(f))

        p = np.exp(F_y_i) / znamenatel #  это софтмакс функция

        loss +=  -np.log(p) # формула кросс-энтропии

        # лямбда-функция
        lambda_func = lambda k: np.exp(f[k]) / znamenatel

        # Для каждого класса вычисляем вероятность (софтмакс функцию) и вычисляем градиент
        for k in range(num_classes):
              p_k = lambda_func(k)
              dW[:, k] += (p_k - (k == y[i])) * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    - X: массив numpy формы (N, D), содержащий минибатч данных.
    - y: массив numpy формы (N,), содержащий обучающие метки; y[i] = c означает что X[i] имеет метку c, где 0 <= c < C.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0] # кол-во выборок

    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)

    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f) / sum_f # софтмакс функция

    # loss = -log(p) для каждого i-й выборки. Здесь делаем чтоб для всех
    loss = np.sum(-np.log(p[np.arange(num_train), y])) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    # Градиент для класса k  i_го образца = p - 1{y_i=k}*x, где 1 индикатор внутреннего условия
    indicate = np.zeros_like(p)

    # y столбец и num_train строки делаем единицами
    indicate[np.arange(num_train), y] = 1 # индикатор правильного класса

    Q = p - indicate # матрца Q строки - размер выборки  \ стобцы - кол-во классов?
    X = X.transpose() # транспонируем
    dW = X.dot(Q) / num_train
    dW += reg * W

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
