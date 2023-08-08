import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Conv1D, LSTM, GRU, MaxPooling1D, Flatten, Dropout
from sklearn.svm import SVR

class My_BPNN():
    def __init__(self):
        self.name = 'BPNN'

    # 定義sigmoid函數
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 前向傳播函數
    def forward(self, X, W1, W2):
        hidden = self.sigmoid(np.dot(X, W1))
        output = self.sigmoid(np.dot(hidden, W2))
        return hidden, output

    # 反向傳播函數
    def backward(self, X_train, Y_train, output, hidden, W2):
        error_output = Y_train - output
        delta_output = error_output * output * (1 - output)
        error_hidden = np.dot(delta_output, W2.T)
        delta_hidden = error_hidden * hidden * (1 - hidden)
        dW2 = np.dot(hidden.T, delta_output)
        dW1 = np.dot(X_train.T, delta_hidden)
        return dW1, dW2

    # 訓練模型函數
    def train(self, X_train, Y_train, epochs, learning_rate):
        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        num_input = X_train.shape[1]
        num_output = Y_train.shape[1]
        num_hidden = 3
        # 初始化權重
        W1 = np.random.randn(num_input, num_hidden)
        W2 = np.random.randn(num_hidden, num_output)
        # 迭代訓練
        for epoch in range(epochs):
            hidden, output = self.forward(X_train, W1, W2)
            dW1, dW2 = self.backward(X_train, Y_train, output, hidden, W2)
            W1 += learning_rate * dW1
            W2 += learning_rate * dW2
            return W1, W2

    # 測試模型函數
    def predict(self, X_test, W1, W2):
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
        _, Y_predict = self.forward(X_test, W1, W2)
        return Y_predict


class My_SVM():
    def __init__(self):
        self.name = 'SVM'
        self.model = SVR(kernel='rbf', C=1, epsilon=0.01)
    
    def train(self, X_train, Y_train):
        # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        self.model.fit(X_train, Y_train)
        return self.model


class My_CNN():
    def __init__(self, steps, features):
        self.name = 'CNN'
        dropout_rate = 0.25
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(steps, features)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(features))

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(X_test, Y_test)) # 以測試資料為驗證資料   
        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        return self.model


class My_LSTM():
    def __init__(self, steps, features):
        self.name = 'LSTM'
        dropout_rate = 0.25
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, input_shape=(steps, features)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(64))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(features))

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(X_test, Y_test)) # 以測試資料為驗證資料   
        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        return self.model


class My_GRU():
    def __init__(self, steps, features):
        self.name = 'GRU'
        dropout_rate = 0.25
        self.model = Sequential()
        self.model.add(GRU(64, return_sequences=True, input_shape=(steps, features)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(64, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(64, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(64))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(features))

    def train(self, X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=True):
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(X_test, Y_test)) # 以測試資料為驗證資料   
        if show_training_history:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        return self.model
    

# *********
# Example *
# *********
# --------------------------------------------------------------------------
# BPNN
# --------------------------------------------------------------------------
# # 傳統方法的參考資料須調整資料維度
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
# Y_train = Y_train.ravel()
# Y_test = Y_test.ravel()

# # 創建模型
# model = My_BPNN()

# # 訓練模型
# W1, W2 = model.train(X_train, Y_train, epochs, lr)

# # 預測
# Y_train_predict = model.predict(X_train, W1, W2)
# Y_predict = model.predict(X_test, W1, W2)

# --------------------------------------------------------------------------
# SVM
# --------------------------------------------------------------------------
# # 傳統方法的參考資料須調整資料維度
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
# Y_train = Y_train.ravel()
# Y_test = Y_test.ravel()

# # 創建模型
# model = My_SVM()

# # 訓練模型
# model = model.train(X_train, Y_train)

# # 預測
# Y_train_predict = model.predict(X_train)
# Y_predict = model.predict(X_test)

# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
# # 創建模型
# model = My_CNN(steps, features)

# # 訓練模型
# model = model.train(X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size)

# # 儲存模型
# model.save(weights_path)
# print('Training weights saved in ' + weights_path)

# # 預測
# Y_train_predict = model.predict(X_train) # 以訓練資料預測
# Y_predict = model.predict(X_test)

# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------
# # 創建模型
# model = My_LSTM(steps, features)

# # 訓練模型
# model = model.train(X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size)

# # 儲存模型
# model.save(weights_path)
# print('Training weights saved in ' + weights_path)

# # 預測
# Y_train_predict = model.predict(X_train) # 以訓練資料預測
# Y_predict = model.predict(X_test)

# --------------------------------------------------------------------------
# GRU
# --------------------------------------------------------------------------
# # 創建模型
# model = My_GRU(steps, features)

# # 訓練模型
# model = model.train(X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size)

# # 儲存模型
# model.save(weights_path)
# print('Training weights saved in ' + weights_path)

# # 預測
# Y_train_predict = model.predict(X_train) # 以訓練資料預測
# Y_predict = model.predict(X_test)