import pickle
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential #在使用前需要先提前导入这个函数
from tensorflow.keras import initializers

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

# 定义模型
def baseline_model():
    model = Sequential()
    model.add(Conv2D(filters=1,
                    kernel_size=(3,3),
                    input_shape = (28,28,1),
                    padding='same',
                    # kernel_initializer= initializers.Ones,
                    activation = 'relu'))
    # model.add(MaxPooling2D(pool_size = (2,2))) 先不加池化
    model.add(Flatten())
    model.add(Dense(mid_layer, activation='relu', use_bias=False))
    model.add(Dense(categories, activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# 训练控制(实验中就算设置epoch为1，batchsize也为1，图片只有一张图片，fit后各层矩阵的秩都是满秩的？框架中随机初始化权重的时候就是满秩进行初始化的么)
def rank_test():
    for i in range(500):
        model.fit(X_train[0:i+1],y_train[0:i+1],validation_data=(X_test[0:i+1],y_test[0:i+1]),epochs=epoch_num,batch_size=1,verbose=2,callbacks=[cbk])
        for j in range(epoch_num):
            my_weights_array1 = np.array(cbk.weights1[j])
            my_weights_array2 = np.array(cbk.weights2[j])
            my_weights1 = np.matrix(my_weights_array1.reshape(784, 200))
            my_weights2 = np.matrix(my_weights_array2.reshape(200, 10))
            print(np.linalg.matrix_rank(my_weights1))
            print(np.linalg.matrix_rank(my_weights2))
    print(my_weights1)
    print(my_weights2)
    print(np.linalg.matrix_rank(my_weights1))
    print(np.linalg.matrix_rank(my_weights2))
    # s,v,d分解
    U,s,V = np.linalg.svd(my_weights1)
    U2,s2,V2 = np.linalg.svd(my_weights2)
    print(U2.shape)
    print(s2.shape)
    print(V2.shape)
    print(np.linalg.matrix_rank(U2))
    print(np.linalg.matrix_rank(s2))
    print(np.linalg.matrix_rank(V2))

# 整个数据集训练
def rank_test_train_finish():
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epoch_num,batch_size=200,verbose=2, callbacks=[cbk]) 
    for j in range(epoch_num):
            my_weights_array1 = np.array(cbk.weights1[j])
            my_weights_array2 = np.array(cbk.weights2[j])
            my_weights1 = np.matrix(my_weights_array1.reshape(784, 200))
            my_weights2 = np.matrix(my_weights_array2.reshape(200, 10))
            print('epoch:',j)
            print(my_weights1)
            print(my_weights2)
            print('第一层权重矩阵秩为：',np.linalg.matrix_rank(my_weights1))
            print('第二层权重矩阵秩为：',np.linalg.matrix_rank(my_weights2))
            # s,v,d分解
            U,s,V = np.linalg.svd(my_weights1)
            U2,s2,V2 = np.linalg.svd(my_weights2)
            
            print('U.shape:',U.shape)
            print('s.shape:',s.shape)
            print('V.shape:',V.shape)
            print('U.rank:',np.linalg.matrix_rank(U))
            print('s.rank:',np.linalg.matrix_rank(s))
            print('V.rank:',np.linalg.matrix_rank(V))
            print('U2.shape:',U2.shape)
            print('s2.shape:',s2.shape)
            print('V2.shape:',V2.shape)
            print('U2.rank:',np.linalg.matrix_rank(U2))
            print('s2.rank:',np.linalg.matrix_rank(s2))
            print('V2.rank:',np.linalg.matrix_rank(V2))


if __name__ == "__main__":
    seed = 7 # 设置随机种子
    categories = 10
    mid_layer = 200
    np.random.seed(seed)
    (X_train,y_train),(X_test,y_test) = mnist.load_data() # 加载数据
    # (X_train,y_train),(X_test,y_test) = cifar10.load_data() # 加载数据
    num_pixels = X_train.shape[1] * X_train.shape[2] # mnist 28 * 28
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
    epoch_num = 10

    model = baseline_model()
    #model.fit() 函数每个参数的意义参考：https://blog.csdn.net/a1111h/article/details/82148497
    cbk = CollectWeightCallback(layer_index1=2, layer_index2=3)
    # model = load_model(r'C:\Users\kirin\Desktop\weight_experiment\my_model.h5')
    # print(numpy.array(cbk.weights[0]).reshape(10,200))
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    # rank_test()
    rank_test_train_finish()
    # model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epoch_num,batch_size=200,verbose=2, callbacks=[cbk]) 
    # 1、模型概括打印
    model.summary()
    # model.save(r'my_model_conv_minst.h5') # 保存模型参数
    # model.save(r'my_model_cifar10.h5')

    # 保存cbk对象
    # global weight_matrix_save
    # weight_matrix_save = pickle.dumps(w) # 报错 typeerror:can't pickle_thread.lock objects
    # with open ("weightinfo","ab") as f:
    #     f.write(weight_matrix_save)
    
    scores = model.evaluate(X_test,y_test,verbose=0) #model.evaluate 返回计算误差和准确率
    print(scores)
    print("Base Error:%.2f%%"%(100-scores[1]*100))

    print(" ")




