#!/usr/bin/env python
# coding: utf-8

# # Day2 exercise （データの水増し，回帰）

# ## MNISTデータから画像の水増し（明度を変える，フリップ，回転，シフト等）を行い，この拡張データを使用して学習を行う．元の学習データに比べ適合率などは変化するかを確認する．また，各自の手書き数字が正しく認識されるかを確認せよ．正しく認識されない場合，正しく認識させるにはどうすれば良いかを考えて試す．

# ## 1. MNISTデータをtensorflowから読み込む

# In[ ]:


from sklearn import datasets
ds_digits= datasets.load_digits()
import numpy as np

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# データ数が多いので一部のみを使用する
n = 600
x_train = x_train[:len(x_train)//n]
y_train = y_train[:len(y_train)//n]
x_test = x_test[:len(x_test)//n]
y_test = y_test[:len(y_test)//n]

print("学習用データ数：", len(x_train))
print("テスト用データ数：", len(x_test))
print("データ形状:", x_train[0].shape)


# ## x_trainの最初のデータを画面に画像として出力する．

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))
plt.imshow(x_train[0],cmap="gray")
plt.show()


# ## x_train画をいくつか出力してデータを確認する．

# In[ ]:


import matplotlib.pyplot as plt
w = 8
h = 4
l = w*h
fig, ax = plt.subplots(h,w,figsize=(4*w, 4*h))
k = 0
for i in range(h):
    for j in range(w):
        ax[i,j].imshow(x_train[k], cmap="gray")
        k += 1
plt.show()


# ## 2. 画像の水増し

# In[ ]:


import numpy as np
from skimage.filters import rank

count = 0
x_tmp = x_train.copy()
y_tmp = y_train.copy()
print(len(x_train))
for img, label in zip(x_tmp, y_tmp):
    aug = [img]
    # 5x5で平滑化
    normal_result = rank.mean(x_train[0], selem=np.ones((5, 5)))
    aug.append(normal_result)
    # 左右フリップ
    aug.append(np.fliplr(img).copy())
    # 上下フリップ
    aug.append(np.flipud(img).copy())
    # 90度回転
    aug.append(np.rot90(img, 1).copy())
    # 180度回転
    #aug.append(np.rot90(img, 2).copy())
    # 270度回転
    aug.append(np.rot90(img, 3).copy())
    # シフト
    shift = 1
    for i in range(-shift, shift+1):
        if i == 0:
            continue
        # 行シフト
        aug.append(np.roll(x_train[0], i, axis=0))
        # 列シフト
        aug.append(np.roll(x_train[0], i, axis=1))
    
    # 学習用データの水増しデータを追加する
    for i in aug:
        x_train = np.vstack((x_train, i.reshape(1, 28, 28)))
        y_train = np.append(y_train, label)
    count += 1

print(len(x_train))


# ## 3. 分類

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Reshape

#　データを正規化する
x_train, x_test = x_train / 255.0, x_test / 255.0 

# モデルを構築する
model = tf.keras.models.Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation='relu'),
  Dropout(0.2),
 Dense(10, activation='softmax')
])

# 学習設定
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの構成を確認する
model.summary()

# モデルの構成をファイルに出力する
tf.keras.utils.plot_model(model, "tensorflow-2021210-aug.pdf")

# エポック数5で学習（途中経過を出力するため「verbose=2」）
history = model.fit(x_train, y_train, epochs=30, verbose=0)

# 評価（評価結果の詳細を表示するため「verbose=2」）
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('Test accuracy: {0:.3f} '.format(test_acc))
print('Test loss: {0:.3f}'.format(test_loss))

# モデル評価
import pandas as pd

results = pd.DataFrame(history.history)
results.plot()


# In[ ]:


# 画像のシフト（ロール）
import numpy as np

tmp = np.roll(x_train[0], -5, axis=0)
print(id(x_train[0]))
print(id(np.roll(x_train[0], -5, axis=0)))

import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))
plt.imshow(tmp, cmap="gray")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

w = 4
h = int(np.ceil(len(aug)/w))
l = len(aug)
fig,ax = plt.subplots(h,w,figsize=(4*w, 4*h))
k = 0
flag = False
for i in range(h):
    for j in range(w):
        if (l-1) <= k:
            flag = True
            break
        ax[i,j].imshow(aug[k], cmap="gray")
        k += 1
    if flag == True:
        break
#plt.show()
plt.savefig('mnist_augumentation.pdf')


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))
plt.imshow(tmp, cmap="gray")
plt.show()


# In[ ]:


hist, bins = np.histogram( x_train[0].flatten(), bins=256 )

# 0〜256までplot
plt.plot( hist )
plt.xlim(0, 256)
print(hist)


# In[ ]:


# 平滑化
import matplotlib.pyplot as plt
from skimage.filters import rank

normal_result = rank.mean(x_train[0], selem=np.ones((5, 5)))

plt.figure(figsize=(4,4))
plt.imshow(normal_result,cmap="gray")
plt.show()


# ## 気象庁のWebページ ``https://www.data.jma.go.jp/gmd/risk/obsdl/index.php`` からアメダス観測点のデータをダウンロードし，観測点の気温を予測せよ．観測点はどこでも良い．
#    
#    - ダウンロードできるデータには，降水量の日合計，日平均気圧，日平均相対湿度，日平均気温，日照時間等がある．これらから「日平均気温」の予測を行う．
#    - どのデータを使用するのが良いか．
#    - どのモデルを使用すれと精度が良いか．

# ## 1. データのダウンロード
# 群馬県桐生市の2010年1月1日から2021年12月9日までのデータをダウンロードした．長い期間のデータを取得したいため，項目毎に別ファイルに保存した．
# - 日照時間.csv: Shift_JIS
# - 日平均気温.csv: Shift_JIS
# - 日平均風速.csv: Shift_JIS
# - 日最深積雪.csv: Shift_JIS
# - 日平均蒸気圧.csv: Shift_JIS
# - 日平均相対湿度.csv: Shift_JIS
# - 降水量の日最大.csv: Shift_JIS
# - 日合計全天日射量.csv: CP932

# ## 2. 文字コードをUTF-8に変換する．
# コマンドnkfを使用すると「　``nkf -w --overwrite -wd　*.csv`` 」で変換できる．コマンドnkfがなければテキストエディタ等を使用して変換を行う．

# In[ ]:





# ## ダウンロードした気象データを読み込む．

# In[ ]:


import pandas as pd
import math
df = pd.read_csv("./data/weather/data気温.csv", encoding="utf-8")
print(df.describe())
train_data = df[df["年"] <= 2019]
pre_data = df[df["年"] >= 2020]
# 直前の6日前のデータから気温を予測する
interval = 6

# 前処理
train_data.dropna(how='any')
pre_data.dropna(how='any')


def make_data(data):
    print(data)
    x = []
    y = []
    temps = list(data["平均気温(℃)"])
    print("make_data:", len(temps))
    
    for i in range(len(temps)):
        if i < interval:
            continue
        if math.isnan(temps[i]) or math.isinf(temps[i]):
            print("***** NAN: ", temps[i])
            y.append(temps[i-1])
        else:
            y.append(temps[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            if math.isnan(temps[d]) or math.isinf(temps[d]):
                xa.append(temps[d-1])
            else:
                xa.append(temps[d])
        x.append(xa)
    
    return x, y

# 学習用データを作成する
x_train, y_train = make_data(train_data)
#print(x_train)
#print(y_train)

# 学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
coef = list(model.coef_)
intercept = model.intercept_

print("回帰係数: ", coef)
print("誤差（切片）: ", intercept)

# 予測
x_test, y_test = make_data(pre_data)
y_pred= model.predict(x_test)

# 平均平方二乗誤差
import numpy as np
rmse_err = np.sqrt(np.mean(np.square(y_pred - y_test)))
print('RMSE error（平均平方二乗誤差率）:', rmse_err)

# プロット
import matplotlib.pyplot as plt


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15,5))
plt.plot(range(len(y_test)), y_test, label="test")
plt.plot(range(len(y_pred)), y_pred, label="pred")
plt.legend()
plt.show()


# In[ ]:




