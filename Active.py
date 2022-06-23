import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv3D,MaxPooling3D,Dropout,BatchNormalization,concatenate,Activation
from tensorflow.keras.utils import to_categorical
import keras

tf.keras.backend.set_image_data_format('channels_first')

path = "./KTH"  #KTH数据集有6类动作，分别是拍手、慢跑、跑、走、挥手和拳击
vf = tf.__version__, keras.__version__
print(vf)
'''
从每个视频中提取9 frames；
对这9个frames 提取5种通道图像；
通道：9个grayscale；9个gradient x；9个gradient y；8个optflow x；8个optflow y；
总共 43 frames。
'''
#定义图像裁剪大小以及提取的帧数
img_rows,img_cols,img_depth=80,60,9
#定义量来存储整个数据集
X_tr=[]

folders = os.listdir(path)
for folder in folders:
    if 'zip' not in folder:
        filepath = os.path.join(path, folder)
        files = os.listdir(filepath)
        for file in files:
            vid = os.path.join(path, folder, file)
            frames1 = []# 保存灰度通道
            frames2 = []# 保存梯度通道 x
            frames3 = []# 保存梯度通道 y
            optflow = np.zeros((img_depth - 1, img_rows, img_cols, 2))
            optflow_x = np.zeros((img_depth - 1, img_rows, img_cols, 2))
            optflow_y = np.zeros((img_depth - 1, img_rows, img_cols, 2))
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5)
            # print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            for k in range(img_depth): # 提取9帧
                ret, frame = cap.read()
                frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA)# 裁剪
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度通道
                gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) # 梯度通道
                gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5) # 梯度通道
                if k + 1 < img_depth:
                    optflow[k] = cv2.calcOpticalFlowFarneback(gray[k], gray[k + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) # 光流通道

                frames1.append(gray)
                frames2.append(gradient_x)
                frames3.append(gradient_y)
                # plt.imshow(gray, cmap = plt.get_cmap('gray'))
                # plt.xticks([]), plt.yticks([])  # 隐藏X轴和Y轴上的刻度值
                # plt.show()
                # cv2.imshow('frame',gray)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            optflow_x = cv2.normalize(optflow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            optflow_x = optflow_x.astype('uint8')
            optflow_y = cv2.normalize(optflow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            optflow_y = optflow_y.astype('uint8')
            f1 = np.array(frames1)
            f2 = np.array(frames2)
            f3 = np.array(frames3)
            f4 = np.array(optflow_x)
            f5 = np.array(optflow_y)
            f4 = np.reshape(f4, (img_depth - 1, img_cols, img_rows))
            f5 = np.reshape(f5, (img_depth - 1, img_cols, img_rows))

            concat = np.concatenate((f1, f2, f3, f4, f5))
            input = np.array(concat)
            # print (input.shape)
            ipt = np.rollaxis(np.rollaxis(input, 2, 0), 2, 0)
            # print (ipt.shape)

            X_tr.append(ipt)

print(len(X_tr))

# 将特征转换为数组
X_tr_array = np.array(X_tr)
X_tr_array = np.reshape(X_tr_array,(599,80,60,43))
# print(X_tr_array.shape)

# 给每个类定义标签
num_samples = len(X_tr_array)
label=np.ones((num_samples,),dtype = int)
label[0:100]= 0
label[100:200] = 1
label[200:300] = 2
label[300:400] = 3
label[400:499]= 4
label[499:] = 5

# 创建数据集
train_data = [X_tr_array, label]
(X_train, y_train) = (train_data[0], train_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows, img_cols, 43))  # 定义单通道
for h in range(num_samples):
    train_set[h][0][:][:][:] = X_train[h, :, :, :]

patch_size = 43  # Img_depth或每个视频使用的帧数

print('train samples:', train_set.shape)

# CNN训练参数
batch_size = 2
nb_classes = 6  #将KTH数据集分为 6 类
nb_epoch = 50

# 将类向量转换为二元类矩阵
Y_train = to_categorical(y_train, nb_classes)

# 预处理
train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /= np.max(train_set)

# 定义模型，应用 3 个 Conv层 和 2 个 maxpool层
model = Sequential()
# 第一层Conv核大小为 9x7x3， 随后是 3x3x1 maxpool
model.add(Conv3D(8, (9, 7, 3),input_shape=(1, img_rows, img_cols, 43), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 1)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
# 第二层Conv核大小为 7x7x3，随后是 3x3x1 maxpool
model.add(Conv3D(16, (7, 7, 3),input_shape=(1, img_rows, img_cols, 43), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3,1)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
# 第三层Conv核大小为 6x4x1，随后是一个 dense layer
model.add(Conv3D(32, (6, 4, 1),input_shape=(1, img_rows, img_cols, 43), activation='relu'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())

#model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(nb_classes, kernel_initializer='normal'))
model.add(Activation('softmax'))    #relu 用作所有层的激活函数，除了最后一层使用 softmax

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


# 划分数据集
X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),batch_size=batch_size,epochs = nb_epoch,shuffle=True)

 # 评估模型
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# 可视化结果
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
# print(plt.style.available)
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()