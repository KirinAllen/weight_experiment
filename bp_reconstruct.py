# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 调用GPU

import csv
import codecs

import cv2
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
# 这里要注意在用from导包的时候要把tensorflow名字写全
from tensorflow.keras.datasets import mnist # 从keras的datasets中导入mnist数据集
from tensorflow.keras.datasets import cifar10 # 从keras的datasets中导入cifar10数据集
from tensorflow.keras.models import Sequential # 导入Sequential模型
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense # 全连接层用Dense类
#from tensorflow.keras.layers import Dropout # 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合
# np_utils在tensorflow.keras.utils没有，在tensorflow.python.keras.utils中（遇到此问题可去相应目录下查看包组织结构）
from tensorflow.python.keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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


# 搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络
def baseline_model():
    model = Sequential() # 建立一个Sequential模型,然后一层一层加入神经元
    # 第一步是确定输入层的数目正确：在创建模型时用input_dim参数确定。例如，有784个个输入变量，就设成num_pixels。
    #全连接层用Dense类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（uniform），Keras的默认方法也是这个。也可以用高斯分布进行初始化（normal）。
    # 具体定义参考：https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch7-develop-your-first-neural-network-with-keras.html
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu', use_bias=False))
    model.add(Dense(first_layer,input_dim=num_pixels,kernel_initializer='normal',activation='relu',use_bias=False))
    model.add(Dense(num_pixels,kernel_initializer='normal',activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


if __name__ == "__main__":
    seed = 7 #设置随机种子
    first_layer = 200
    numpy.random.seed(seed)
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
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    # num_classes = y_test.shape[1]
    epoch_num = 5
    model = baseline_model()
    # 1、模型概括打印
    model.summary()
    #model.fit() 函数每个参数的意义参考：https://blog.csdn.net/a1111h/article/details/82148497
    cbk = CollectWeightCallback(layer_index1=1, layer_index2=2)
    # model = load_model(r'C:\Users\kirin\Desktop\weight_experiment\my_model.h5')
    # print(numpy.array(cbk.weights[0]).reshape(10,200))
    model.fit(X_train,X_train,validation_data=(X_test,X_test),epochs=epoch_num,batch_size=200,verbose=2, callbacks=[cbk]) 
    
    # model.save(r'my_model_minst.h5')
    # model.save(r'my_model_cifar10.h5')
    layer = model.get_layer(index=2)
    output = layer.reshape(28,28)

    scores = model.evaluate(X_test,y_test,verbose=0) #model.evaluate 返回计算误差和准确率
    print(scores)
    print("Base Error:%.2f%%"%(100-scores[1]*100))

    # 基础实验
    # num_w = numpy.random.randint(0,784,size=20)
    # for i in range(20):
    #     show_each_weight(num_w[i])
    # show_weight()
    # cv2.destroyAllWindows()

