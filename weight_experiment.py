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

# # 定义重写回调类获取各层权重
# class CollectWeightCallback(Callback):
#     def __init__(self, layer_index):
#         super(CollectWeightCallback, self).__init__()
#         self.layer_index = layer_index
#         self.weights = []
    
#     def on_epoch_end(self, epoch, logs=None):
#         layer = self.model.layers[self.layer_index]
#         self.weights.append(layer.get_weights())

# 把训练结果权值保存到csv文件中
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

# 把训练结果保存为txt文件
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功") 


# 搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络
def baseline_model():
    model = Sequential() # 建立一个Sequential模型,然后一层一层加入神经元
    # 第一步是确定输入层的数目正确：在创建模型时用input_dim参数确定。例如，有784个个输入变量，就设成num_pixels。
    #全连接层用Dense类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（uniform），Keras的默认方法也是这个。也可以用高斯分布进行初始化（normal）。
    # 具体定义参考：https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch7-develop-your-first-neural-network-with-keras.html
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu', use_bias=False))
    model.add(Dense(first_layer,input_dim=num_pixels,kernel_initializer='normal',activation='relu',use_bias=False))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

 
# 可视化权重矩阵
def show_weight():
    for i in range(epoch_num):
        my_weights_array = numpy.array(cbk.weights[i])
        # my_weights_expand = my_weights_array.reshape(10,200).repeat(100,axis=0).repeat(5,axis=1)
        # my_weights = numpy.matrix(my_weights_array.reshape(784, 200)) # minst 第一层
        my_weights = numpy.matrix(my_weights_array.reshape(3072, 200)) #cifar10 第一层
        # my_weights_expand = numpy.matrix(numpy.zeros((1000,1000)))

        print(my_weights)
        print(numpy.linalg.matrix_rank(my_weights)) # 获取每个矩阵的秩
        cv2.imshow('e'+str(i)+'_w', my_weights*100)
        # cv2.imwrite('C:/Users/kirin/Desktop/weight_experiment/img'+str(i)+'.jpg', my_weights)
        cv2.waitKey(0)
        plt.plot(my_weights[:,0:200])
        plt.show()

# 每个权重随着epoch的变化趋势
def list_weight(w_num_row=None, w_num_column=0):
    weight = []
    for i in range(epoch_num):
        weight_arrays = cbk.weights[i]
        # weight_matrix = numpy.matrix(weight_arrays.reshape(200,10))
        weight.append(weight_arrays[0][w_num_row][w_num_column])
    return weight

def show_each_weight(weight_sub=None):
    x_list = []
    for i in range(epoch_num):
        x_list.append(i)
    w = list_weight(w_num_row=weight_sub)
    plt.figure('w'+str(weight_sub))
    plt.plot(x_list, w, color='r', linewidth=1,alpha=0.6)
    plt.show()


if __name__ == "__main__":
    seed = 7 #设置随机种子
    first_layer = 200
    second_layer = 10
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
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    epoch_num = 5
    model = baseline_model()
    #model.fit() 函数每个参数的意义参考：https://blog.csdn.net/a1111h/article/details/82148497
    cbk = CollectWeightCallback(layer_index1=1, layer_index2=2)
    # model = load_model(r'C:\Users\kirin\Desktop\weight_experiment\my_model.h5')
    # print(numpy.array(cbk.weights[0]).reshape(10,200))
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epoch_num,batch_size=200,verbose=2, callbacks=[cbk]) 
    # 1、模型概括打印
    model.summary()
    model.save(r'my_model_minst.h5')
    # model.save(r'my_model_cifar10.h5')
    scores = model.evaluate(X_test,y_test,verbose=0) #model.evaluate 返回计算误差和准确率
    print(scores)
    print("Base Error:%.2f%%"%(100-scores[1]*100))

    # 基础实验
    num_w = numpy.random.randint(0,784,size=20)
    for i in range(20):
        show_each_weight(num_w[i])
    show_weight()
    cv2.destroyAllWindows()

