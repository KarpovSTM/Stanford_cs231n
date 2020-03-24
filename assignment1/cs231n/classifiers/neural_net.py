from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.

    Двухслойная полностью связная нейронная сеть. Сеть имеет входное измерение N,
    скрытое измерение уровня H, и выполняет классификацию по классам C.
     Мы обучаем сеть с функцией потерь softmax и регуляризацией L2 на весовых матрицах.
    Сеть использует нелинейность ReLU после первого полностью подключенного уровня.

     Другими словами, сеть имеет следующую архитектуру:

     вход - полностью связанный слой - ReLU - полностью связанный слой - softmax

     Выходы второго полностью связного слоя - это баллы для каждого класса
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.

        Инициализируйте модель. Веса инициализируются небольшими случайными значениями,
         а смещения инициализируются нулями. Веса и смещения хранятся в
         переменной self.params, которая представляет собой словарь со следующими ключами:

         W1: вес первого слоя; имеет форму (D, H)
         b1: смещения первого слоя; имеет форму (H,)
         W2: веса второго слоя; имеет форму (H, C)
         b2: смещения второго слоя; имеет форму (С,)

         Входы:
         - input_size: размерность D входных данных.
         - hidden_size: количество нейронов H в скрытом слое.
         - output_size: количество классов C.

        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.

          Вычислить потери и градиенты для двухслойной полностью подключенной нейронной сети.

         Входы:
         - X: входные данные формы (N, D). Каждый X [i] является тренировочным образцом.
         - y: вектор обучающих меток. y [i] является меткой для X [i], и каждый y [i]
        является целым числом в диапазоне 0 <= y [i] <C. Этот параметр является необязательным;
        если он не пройден, мы возвращаем только результаты, а если он пройден, мы вместо этого возвращаем потери и градиенты.
         - рег: сила регуляризации.

         Возвращает:
         Если y равен None, вернуть матричные оценки формы (N, C), где оценка [i, c] -
         оценка для класса c на входе X [i].

         Если y не None, вместо этого верните кортеж:
         - потеря: потеря (потеря данных и потеря регуляризации) для этой партии обучающих выборок.
         - grads: словарь отображает имена параметров в градиенты этих параметров относительно функции потерь; имеет те же ключи, что и self.params.


        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        """
        TODO: выполнить прямой проход, вычисляя вероятности классов для ввода. #
        Сохраните результат в переменной scores, которая должна быть массивом формы (N, C).
        """
        #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # первый слой первый проход
        scores1 = X.dot(W1) + b1

        Relu = np.maximum(0, scores1) # возвращает все значения scores1 больше нуля (Relu интерпретация)

        scores = Relu.dot(W2) + b2 # второй слой, вторая матрица весов и смещение

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        """
        Закончите прямой проход и рассчитайте потери. Это должно включать как потерю данных, так и регуляризацию L2 для W1 и W2.
        Сохраните результат в переменной loss, которая должна быть скалярной. Используйте классификатор потерь Softmax.
        """
        #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # получаем ненормированные вероятности
        exp_scores = np.exp(scores)
        
        # нормализуем значения
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Теперь у нас есть массив probs размером [300 x 3], где каждая строка теперь содержит класс вероятностей.
        # В частности, поскольку мы нормализовали их, каждая строка теперь суммируется в одну.

        # Одномерный массив, состоящий только из вероятностей, назначенных правильным классам для каждого примера.
        correct_logprobs = -np.log(probs[range(N),y])

        # compute the loss: average cross-entropy loss and regularization
        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        """
        Вычислить обратный проход, вычисляя производные от весов и смещений.
        Сохраните результаты в словаре оценок.
        Например, grads ['W1'] должен хранить градиент на W1 и быть матрицей того же размера #
        """
        #############################################################################

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # градиент вероятностей
        dscores = probs # probs хранит вероятности всех классов (в виде строк) для каждого примера
        dscores[range(N),y] -= 1
        dscores /= N

        # backprop через матричное умножение

        # 2-й слой
        grads['W2'] = np.dot(Relu.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        grads['W2'] += reg * W2

        dhidden = np.dot(dscores, W2.T)

        # backprop Relu скрытый слой
        dhidden[Relu <= 0] = 0

        # 1-й слой
        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)
        grads['W1'] += reg * W1

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.

        Входы:
         - X: массив фигур (N, D), дающий тренировочные данные.
         - y: форма массива n (n,), дающая метки обучения; y [i] = c означает, что
           X [i] имеет метку c, где 0 <= c <C.
         - X_val: массивный массив формы (N_val, D), предоставляющий данные проверки.
         - y_val: массивный массив формы (N_val,), дающий метки проверки.
         - learning_rate: скалярная скорость обучения для оптимизации.
         - learning_rate_decay: скалярный коэффициент, используемый для снижения скорости обучения
           после каждой эпохи.
         - reg: Скаляр, дающий силу регуляризации.
         - num_iters: количество шагов при оптимизации.
         - batch_size: количество обучающих примеров для использования на шаг.
         - подробный: логический; если true, то печатать прогресс во время оптимизации.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            """
            Создайте случайную мини-партию обучающих данных и меток,
            сохраняя их в X_batch и y_batch соответственно.
            """
            #########################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            sample_index = np.random.choice(np.arange(num_train), batch_size)

            X_batch = X[sample_index]
            y_batch = y[sample_index]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            """
            Использовать градиенты в словаре градов для обновления параметров сети
            (хранящихся в словаре self.params) с использованием стохастического градиентного спуска.
            Вам нужно будет использовать градиенты, хранящиеся в словаре градов, определенном выше.
            """
            #########################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.

        Используйте обученные веса этой двухслойной сети, чтобы предсказать метки для точек данных.
        Для каждой точки данных мы прогнозируем оценки для каждого из классов C
        и присваиваем каждую точку данных классу с наивысшей оценкой.

         Входы:
         - X: бесформенный массив формы (N, D), дающий N D-мерных точек данных для классификации.

         Возвращает:
         - y_pred: пустой массив формы (N,), дающий предсказанные метки для каждого из элементов X.
        Для всех i, y_pred [i] = c означает, что у X [i], как ожидается, будет класс c, где 0 <= с <к.

        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scoreee = X.dot(self.params['W1']) + self.params['b1']

        reluuu = np.maximum(0, scoreee)

        scores = reluuu.dot(self.params['W2']) + self.params['b2']

        y_pred = np.argmax(scores, axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
