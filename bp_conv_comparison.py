# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 调用GPU

import csv
import codecs

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
# 这里要注意在用from导包的时候要把tensorflow名字写全
from tensorflow.keras.datasets import mnist # 从keras的datasets中导入mnist数据集
from tensorflow.keras.datasets import cifar10 # 从keras的datasets中导入cifar10数据集
from tensorflow.keras.models import Sequential # 导入Sequential模型
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense # 全连接层用Dense类
from tensorflow.python.keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# 定义重写回调类获取各层权重
class CollectWeightCallback(Callback):
    def __init__(self, layer_index1, layer_index2):
        super(CollectWeightCallback, self).__init__()
        self.layer_index1 = layer_index1
        self.layer_index2 = layer_index2
        self.weights1 = []
        self.weights2 = []
    
    def on_epoch_end(self, epoch, logs=None):
        layer1 = self.model.layers[self.layer_index1]
        self.weights1.append(layer1.get_weights())
        layer2 = self.model.layers[self.layer_index2]
        self.weights2.append(layer2.get_weights())

# Lemma Two matrices A (nxn) and B (nxn) are similar if and only if the rank of (lamdaI-A)^p equals the rank of (lamdaI-B)^p for any complex number lamda and for
# any integer p ,1 <= p <= n
def matrixs_similarity(A, B, lam):
    print()


# 搭建bp神经网络模型了，创建一个函数，建立含有一个隐层的神经网络
def bp_baseline_model():
    model = Sequential() # 建立一个Sequential模型,然后一层一层加入神经元
    # 第一步是确定输入层的数目正确：在创建模型时用input_dim参数确定。例如，有784个个输入变量，就设成num_pixels。
    #全连接层用Dense类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（uniform），Keras的默认方法也是这个。也可以用高斯分布进行初始化（normal）。
    # 具体定义参考：https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch7-develop-your-first-neural-network-with-keras.html
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu', use_bias=False))
    model.add(Dense(first_layer,input_dim=num_pixels,kernel_initializer='normal',activation='relu',use_bias=False))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# 定义卷积模型
def conv_baseline_model():
    model = Sequential()
    model.add(Conv2D(filters=1,
                    kernel_size=(3,3),
                    input_shape = (28,28,1),
                    padding='same',
                    # kernel_initializer= initializers.Ones,
                    activation = 'relu'))
    # model.add(MaxPooling2D(pool_size = (2,2))) 先不加池化
    model.add(Flatten())
    model.add(Dense(first_layer, activation='relu', use_bias=False))
    model.add(Dense(second_layer, activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# 两个网络的权重比较
def weight_comparison():

    for i in range(epoch_num):
        print('epoch:',i+1)
        # bp 网络
        bp_weights_array1 = np.array(cbk_bp.weights1[i])
        bp_weights_array2 = np.array(cbk_bp.weights2[i])
        bp_weights1 = np.matrix(bp_weights_array1.reshape(784, 200))
        bp_weights2 = np.matrix(bp_weights_array2.reshape(200, 10))
        print('bp网络第一层权重矩阵为：', bp_weights1)
        print('------------------------------------------------------------------------------')
        print('bp网络第二层权重矩阵为:', bp_weights2)
        print('-------------------------------------------------------------------------------')
        bp_eigenvalues1, bp_eigenvectors1 = np.linalg.eigh(bp_weights1.dot(bp_weights1.T))
        bp_eigenvalues2, bp_eigenvectors2 = np.linalg.eigh(bp_weights2.dot(bp_weights2.T))
        # print('bp_weights1 eigenvalues:',bp_eigenvalues1)
        # print('bp_weights1 eigenvectors:',bp_eigenvectors1)
        # print('bp_weights2 eigenvalues:',bp_eigenvalues2)
        # print('bp_weights2 eigenvectors:',bp_eigenvectors2)
        # conv 网络
        conv_weight_array1 = np.array(cbk_conv.weights1[i])
        conv_weight_array2 = np.array(cbk_conv.weights2[i])
        conv_weights1 = np.matrix(conv_weight_array1.reshape(784, 200))
        conv_weights2 = np.matrix(conv_weight_array2.reshape(200, 10))
        print('conv网络第一层权重矩阵为：', conv_weights1)
        print('---------------------------------------------------------------------------------')
        print('conv网络第二层权重矩阵为:', conv_weights2)
        print('---------------------------------------------------------------------------------')
        conv_eigenvalues1, conv_eigenvectors1 = np.linalg.eigh(conv_weights1.dot(conv_weights1.T))
        conv_eigenvalues2, conv_eigenvectors2 = np.linalg.eigh(conv_weights2.dot(conv_weights2.T))
        # print('conv_weights1 eigenvalues:',conv_eigenvalues1)
        # print('conv_weights1 eigenvectors:',conv_eigenvectors1)
        # print('conv_weights2 eigenvalues:',conv_eigenvalues2)
        # print('conv_weights2 eigenvectors:',conv_eigenvectors2)
        print('两个网络第一层特征值之差：', bp_eigenvalues1-conv_eigenvalues1)
        print('两个网络第二层特征值之差：',bp_eigenvalues2-conv_eigenvalues2)
        print('--------------------------------------------------------------------------------')
        print('两个网络第一层特征矩阵之差：',bp_eigenvectors1-conv_eigenvectors1)
        print('两个网络第二层特征矩阵之差：',bp_eigenvectors2-conv_eigenvectors2)
        print('---------------------------------------------------------------------------------')
        print('两个网络第一层权重矩阵之差：',bp_weights1-conv_weights1)
        print('两个网络第二层权重矩阵之差：',bp_weights2-conv_weights2)


if __name__ == "__main__":
    seed = 7 #设置随机种子
    first_layer = 200
    second_layer = 10
    np.random.seed(seed)
    (X_train,y_train),(X_test,y_test) = mnist.load_data() #加载数据
    # (X_train,y_train),(X_test,y_test) = cifar10.load_data() #加载数据
    #print(X_train.shape[0])
    #数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过程。
    num_pixels = X_train.shape[1] * X_train.shape[2] # minst
    # num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3] # cifar10
    
    X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

    #给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encoding
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    epoch_num = 5

    # bp网络训练
    bp_model = bp_baseline_model()
    #model.fit() 函数每个参数的意义参考：https://blog.csdn.net/a1111h/article/details/82148497
    cbk_bp = CollectWeightCallback(layer_index1=1, layer_index2=2)
    bp_model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epoch_num,batch_size=200,verbose=2, callbacks=[cbk_bp]) 
    # bp模型概括打印
    bp_model.summary()
    bp_model.save(r'my_model_minst.h5')
    # model.save(r'my_model_cifar10.h5')
    scores_bp = bp_model.evaluate(X_test,y_test,verbose=0) #model.evaluate 返回计算误差和准确率
    print(scores_bp)
    print("Base Error:%.2f%%"%(100-scores_bp[1]*100))

    # 卷积网络训练
    conv_model = conv_baseline_model()
    cbk_conv = CollectWeightCallback(layer_index1=2,layer_index2=3)
    X_train_conv = X_train.reshape(60000,28,28,1)
    X_test_conv = X_test.reshape(10000,28,28,1)
    conv_model.fit(X_train_conv,y_train,validation_data=(X_test_conv,y_test),epochs=epoch_num,batch_size=200,verbose=2,callbacks=[cbk_conv])
    # conv模型概括打印
    conv_model.summary()
    conv_model.save(r'my_conv_model_minst.h5')
    scores_conv = conv_model.evaluate(X_test_conv,y_test,verbose=0) #model.evaluate返回计算误差和准确率
    print(scores_conv)
    print("Base Error:%.2f%%"%(100-scores_conv[1]*100))
    weight_comparison()

    print(' ')
