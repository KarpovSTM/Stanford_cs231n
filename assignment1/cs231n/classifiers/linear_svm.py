from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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

    Структурированная функция потерь SVM, наивная реализация (с циклами).

     Входы имеют размерность D, есть классы C, и мы работаем на мини-пакетах
     из N примеров.

     Входы:
     - W: массив фигур (D, C), содержащий веса.
     - X: бесформенный массив фигур (N, D), содержащий мини-пакет данных.
     - y: массив фигур (N,), содержащий обучающие метки; у [i] = с означает
       что X [i] имеет метку c, где 0 <= c < C.
     - reg: (float) сила регуляризации

     Возвращает кортеж из:
     - потери с одиночной точностью
     - градиент по отношению к весам W; массив той же формы, что и W

    - x - это вектор-столбец, представляющий изображение (например, 3073 x 1 в CIFAR-10)
     с добавленным измерением смещения в 3073-й позиции (то есть трюк смещения)
   - у - целое число, дающее индекс правильного класса (например, от 0 до 9 в CIFAR-10)
   - W - весовая матрица (например, 10 x 3073 в CIFAR-10)
    """
    dW = np.zeros(W.shape) # инциализируем градиент из нулевых элементов

    # вычислим потери и градиент
    num_classes = W.shape[1] # число классов
    num_train = X.shape[0] # число пикселей изображения
    loss = 0.0
    for i in range(num_train): # для каждого пикселя изображения
        scores = X[i].dot(W) # вероятность становится размером 10 х 1, вероятности для каждого класса
        correct_class_score = scores[y[i]]# сохраняем вероятность истинного класса
        false_count = 0
        for j in range(num_classes):# перебирать все классы
            if j == y[i]: # пропустить истинный класс, чтобы зациклить только неправильные классы
                continue
                # накапливаем потери для i-го примера
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ТУТ ЖОСКА
    #http://cs231n.github.io/optimization-1/#analytic

    # false_count подсчитываем количество классов, которые не достигли требуемого margin (и, следовательно, внесли свой вклад в функцию потерь)
              false_count = false_count + 1
              dW[:, j] = dW[:, j] + X[i] # обновление градиента для ложных классов
              loss = loss + margin
    # обновление градиента для истинных см условие if j == y[i]:
        dW[:, y[i]] = dW[:, y[i]] - false_count * X[i]

    # вычислим среднюю ошибку и градиент по всем
    loss /= num_train
    dW /= num_train

    # добавим регуляризацию
    dW += reg * W
    loss += reg * np.sum(W * W)

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    """
    Вычислить градиент функции потерь и сохранить его в dW.
    Вместо того, чтобы сначала вычислять потери, а затем вычислять производную,
    может быть проще вычислить производную одновременно с вычислением потери.
    В результате вам может понадобиться изменить часть кода выше для вычисления градиента.
    """
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    Решение выше
    """

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    - x - это вектор-столбец, представляющий изображение (например, 3073 x 1 в CIFAR-10)
     с добавленным измерением смещения в 3073-й позиции (то есть трюк смещения)
   - у - целое число, дающее индекс правильного класса (например, от 0 до 9 в CIFAR-10)
   - W - весовая матрица (например, 10 x 3073 в CIFAR-10)
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    """
    Реализуйте векторизованную версию структурированных потерь SVM, сохраняя результат в потерях.
    """
    #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # размер тренировочной выборки (пихели)
    delta = 1.0
    scores = X.dot(W) # вероятности размером 10 х 1 за каждый класс

    # сохраняем вероятность истинного класса для каждого пикселя в вектор
    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score = correct_class_score[:, np.newaxis]

    margins = np.maximum(0, scores - correct_class_score + delta)
    margins[np.arange(num_train), y] = 0 # все ячейки отображающие правильный класс =0

    loss = np.sum(margins)

    loss /= num_train
    loss += reg * np.sum(W * W)

    ################# Градиент ################

    X_vector = np.zeros(margins.shape) # строка х столбец = класс х объект

    X_vector[margins > 0] = 1 # считаем неправильные классы

    incorrect_counts = np.sum(X_vector, axis=1) # сумма стобцов каждой строки, колличество неправильных классов с margin>0

    X_vector[np.arange(num_train), y] = -incorrect_counts # для каждого образца записываем -кол-во неправильных классов

    dW = X.T.dot(X_vector) # матрица градиента

    dW /= num_train
    dW += reg*W
    """
    for i in range(num_train): # для каждоой выборки
        scores = X[i].dot(W) # скор размером 10 х 1 за каждый класс
        correct_class_score = scores[y[i]] # запишем настоящий класс текущего объекта выборки

        margins = np.maximum(0, scores - correct_class_score + delta)
        margins[y[i]] = 0

        loss = np.sum(margins)

        loss /= num_train
        loss += reg * np.sum(W * W)
        """
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    """
    Реализовать векторизованную версию градиента для структурированных потерь SVM,
    сохраняя результат в dW.
    Подсказка: вместо того, чтобы вычислять градиент с нуля, может быть проще
    повторно использовать некоторые промежуточные значения, которые вы
    использовали для вычисления потерь.
    """
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
