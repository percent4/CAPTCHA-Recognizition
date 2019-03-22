import numpy as np
import cv2
import os
import random
from keras.models import *
from keras.layers import *
from keras import callbacks

characters = '0123456789'
width, height, n_len, n_class = 50, 22, 4, 10

def gen(dir, batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    files = os.listdir(dir)
    while True:
        for i in range(batch_size):
            path = random.choice(files)
            imagePixel = cv2.imread(dir+'/'+path, 1)
            filename = path[:4]
            X[i] = imagePixel
            for j, ch in enumerate(filename):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1

        yield X, y





input_tensor = Input((height, width, 3))
x = input_tensor

# 產生有四個 block的卷積網絡
for i in range(4):
    x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)

# 多輸出模型, 由次這個captcha是有4個字母(固定長度), 因此我們對應使用了4個'softmax'來分別預測4個字母的產出
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]

model = Model(inputs=input_tensor, outputs=x)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


cbks = [callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)]

dir = './result'
history = model.fit_generator(gen(dir, batch_size=8),      # 每次生成器會產生32筆小批量的資料
                    steps_per_epoch=120,    # 每次的epoch要訓練30批量的資料
                    epochs=5,                # 總共跑5個訓練循環
                    callbacks=cbks,          # 保存最好的模型到檔案
                    validation_data=gen(dir),   # 驗證資料也是用生成器來產生
                    validation_steps=10      # 用10組資料來驗證
                   )

# 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
import matplotlib.pyplot as plt


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


# 打印整體的loss與val_loss
plot_train_history(history, 'loss', 'val_loss')

plt.figure(figsize=(12, 4))

# 第一個字母的正確率
plt.subplot(2, 2, 1)
plot_train_history(history, 'c1_acc', 'val_c1_acc')

# 第二個字母的正確率
plt.subplot(2, 2, 2)
plot_train_history(history, 'c2_acc', 'val_c2_acc')

# 第三個字母的正確率
plt.subplot(2, 2, 3)
plot_train_history(history, 'c3_acc', 'val_c3_acc')

# 第四個字母的正確率
plt.subplot(2, 2, 4)
plot_train_history(history, 'c4_acc', 'val_c4_acc')

plt.savefig('./train.png')
