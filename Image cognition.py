#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().run_cell_magic('time', '', '1+2')


# In[1]:


get_ipython().run_cell_magic('timeit', '', '1+2')


# In[4]:


start, end = 1, 10
for i in range(start, end):
    if i % 2 == 0:
        print("{0} is even".format(i))
    else:
        print("{0} is odd".format(i))
print('finished')


# In[5]:


get_ipython().run_cell_magic('time', '', "#値の代入\nx = 100\ny = 25.4\nz = 'Hello,World!'\n\nprint(x,y,z)")


# In[18]:


x, y = 5, 3
print(x+y, x-y, x*y, x/y)
print(x%y, x//y, x**y)
print(2**1/2)
a, b = 6, 2
i = 10*(a+4)/b
print(i)


# In[30]:


x = ['orange', 'apple', 'grapes', 'banana']
print(x)
print(x[0])
x[1] = 'lemon'
print(x)
print(x[1:3])
x = ['orange', 'orange']
print(x)


# In[31]:


import math
x = 2
math.sqrt(x)


# In[1]:


x = 0

if x > 0:
    print('1. positive number')
else:
    print('1. others')
    
if x > 0:
    print('2. positive number')
elif x < 0:
    print('2. negative number')
else:
    print('2. zero')
    
x = -1

if x > 0:
    print('3. positive number')
elif x < 0:
    print('3. negative number')
else:
    print('3.zero')


# In[2]:


#x = int(input()) #jupyter notebookの場合

x = 5
if x%5 == 0: # or if not x%5:
    print('5の倍数')
else:
    print('5の倍数ではありません')


# In[5]:


for i in range(5):
    print('Gunma')


# In[6]:


x = 5
while x > 0:
    print(x, end='')
    x -= 1
print()


# In[8]:


x = 1
while x <= 5:
    print(x, end='')
    x += 1
print()


# In[9]:


for x in [1,2,3,4,5]:
    print(x, end='')


# In[10]:


import math
print(math.log(1000))
print(math.sqrt(1000))
print(math.pow(2,3))


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [10,0,20,5,100]

print(x)
print(y)

plt.plot(x, y, marker='o')
plt.show()


# In[38]:


import glob
import numpy as np
from skimage import io

DOG_DIR = './data/img/dog'
CAT_DIR = './data/img/cat'
DOG_LABEL = 0
CAT_LABEL = 1
x = []
y = []

for file in glob.glob(DOG_DIR+'/*.jpg'):
    x.append(np.array(io.imread(file), dtype=np.float16).flatten())
    y.append(DOG_LABEL)
    
for file in glob.glob(CAT_DIR+'/*.jpg'):
    x.append(np.array(io.imread(file), dtype=np.float16).flatten())
    y.append(CAT_LABEL)
    
print("データ数:",len(x))
    
x = np.asarray(x)
y = np.asarray(y)

x /= 255

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test     = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn import svm, metrics

clf = svm.SVC()
clf.fit(x_train, y_train)

pre = clf.predict(x_test)

ac_score = metrics.accuracy_score(y_test, pre)
print('正解率:',ac_score)


# In[48]:


import glob
import numpy as np
from skimage import io

DOG_DIR = './data/img/dog'
CAT_DIR = './data/img/cat'
DOG_LABEL = 0
CAT_LABEL = 1
x = []
y = []

for file in glob.glob(DOG_DIR+'/*.jpg'):
    x.append(np.array(io.imread(file), dtype=np.float16).flatten())
    y.append(DOG_LABEL)
    
for file in glob.glob(CAT_DIR+'/*.jpg'):
    x.append(np.array(io.imread(file), dtype=np.float16).flatten())
    y.append(CAT_LABEL)
    
print("データ数:",len(x))

import matplotlib.pyplot as plt

w = 4
h = 4
fig,ax = plt.subplots(h,w,figsize=(4*w,4*h))
k = 0
for i in range(h):
    for j in range(w):
        ax[i,j].imshow((x[k].reshape([64,64])).astype(int), cmap="gray")
        k += 1
plt.show()
    
x = np.asarray(x)
y = np.asarray(y)

x /= 255

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test     = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn import svm, metrics

clf = svm.SVC()
clf.fit(x_train, y_train)

pre = clf.predict(x_test)

ac_score = metrics.accuracy_score(y_test, pre)
print('正解率:',ac_score)

import matplotlib.image as imread

false_img = x_test[pre != y_test]
print("間違えた画像数:",len(false_img))
w = 8
h = int(np.ceil(len(false_img)/w))
l = len(false_img)#w*h
fig, ax = plt.subplots(h,w,figsize=(4*w,4*h))
k = 0
flag = False
for i in range(h):
    for j in range(w):
        if (l-1) <=k:
            flag = True
            break
        ax[i,j].imshow((false_img[k].reshape([64,64])*255).astype(int),cmap="gray")
        k += 1
    if flag == True:
        break
plt.show()
#plt.savefig('dogcat_flse.pdf')


# In[24]:


import pandas as pd
DATA = './data/NationalFlag/flags_iso.csv'
IMG_DIR = './data/NationalFlag/flagimg'

df = pd.read_csv(DATA)
print(df.head())
print(df.describe())

import os
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
    print("国旗画像を保存するディレクトリを作成しました")
    
    import requests
    for index, rows in df.iterrows():
        country = rows[0]
        url = rows[3]
        file = url.split('/')[-1]
        #print("URL",url)
        #print(os.path.join(IMG_DIR,file))
        f = open(os.path.join(IMG_DIR, file),"wb")
        res = requests.get(url)
        f.write(res.content)
        f.close()
        
from urllib.parse import urlparse

df_new = df.assign(FILE='')
for index, row in df_new.iterrows():
    filename = os.path.basename(urlparse(row[3]).path)
    file = os.path.join(IMG_DIR, filename)
    df_new.at[index, 'FILE'] = file
    
#print(df_new['FILE'])

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

w = 20
h = int(np.ceil(len(df_new)/w))
l = len(df_new) + 1
fig,ax = plt.subplots(h,w,figsize = (4*w,4*h))
k = 0
flag = False
for i in range(h):
    for j in range(w):
        if(l-1) <= k:
            flag = True
            break
        ax[i,j].imshow(imread(df_new.at[k, 'FILE']))
        ax[i,j].set_title(df_new.at[k,'Country'])
        
        k += 1
    if flag == True:
        break
plt.show()

from skimage import io
from skimage.transform import resize
from skimage import color

x = []
y = []

for index, row in df_new.iterrows():
    file = df_new.at[index, 'FILE']
    img = io.imread(file)
    img_resize = resize(img,(80,120),anti_aliasing=True)
    img_gray = color.rgb2gray(color.rgba2rgb(img_resize))
    x.append(np.array(img_gray,dtype=np.float32).flatten())
    y.append(index)
    
x = np.asarray(x)
y = np.asarray(y)

x /= 255

print("データ数:", x.shape)

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

w = 20
h = int(np.ceil(len(df_new)/w))
l = len(df_new) + 1
fig,ax = plt.subplots(h,w,figsize = (4*w,4*h))
k = 0
flag = False
for i in range(h):
    for j in range(w):
        if(l-1) <= k:
            flag = True
            break
        ax[i,j].imshow(x[k].reshape(80, 120), cmap='gray')
        ax[i,j].set_title(df_new.at[k,'Country'])
        
        k += 1
    if flag == True:
        break
plt.show()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

distortion = {}

for i in range(2,100):
    kmeans = KMeans(init='random',n_clusters=i)
    kmeans.fit(x)
    distortion[i] = kmeans.inertia_
    
pd.Series(distortion).plot.line()

from sklearn.cluster import KMeans

n = 10

model = KMeans(init='random',n_clusters=n)
model.fit(x)
y_pred = model.predict(x)

cls = pd.Series(y_pred, name='Class')
df_cls = pd.concat([df_new,cls],axis=1)
df_cls[df_cls['Class'] == 1]

for c in range(n):
    print('Class',c)
    df_i = df_cls[df_cls['Class'] == c]
    print(df_i['Country'].tolist())
    
for c in range(n):
    print('Class',c)
    df_i = df_cls[df_cls['Class'] == c]
    df_country = df_i['Country'].tolist()
    w = 20
    h = int(np.ceil(len(df_i.index)/w))
    l = len(df_i.index) + 1
    fig,ax = plt.subplots(h,w,figsize=(4*w,4*h))
    k = 0
    flag = False
    for i in range(h):
        for j in range(w):
            if (l-1) <= k:
                flag = True
                break
            if h == 1:
                ax[j].imshow(x[df_i.index[k]].reshape(80,120),cmap='gray')
                ax[j].set_title(df_new.at[df_i.index[k],'Country'])
            else:
                ax[i,j].imshow(x[df_i.index[k]].reshape(80,120),cmap='gray')
                ax[i,j].set_title(df_new.at[df_i.index[k],'Country'])
            k += 1
        if flag == True:
            break
plt.show()


# In[ ]:


import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

n = 600
x_train = x_train[:len(x_train)//n]
y_train = y_train[:len(y_train)//n]
x_test = x_test[:len(x_test)//n]
y_test = y_test[:len(y_test)//n]

print("学習用データ数:",len(x_train))
print("テスト用データ数:",len(x_test))
print("データ形状:",x_train[0].shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))
plt.imshow(x_train[0],cmap='gray')
plt.show()

import matplotlib.pyplot as plt
w = 8
h = 4
l = w*h
fig,ax = plt.subplots(h,w,figsize=(4*w,4*h))
k = 0
for i in range(h):
    for j in range(w):
        ax[i,j].imshow(w_train[k],cmap="gray")
        k += 1
plt.show()

import numpy as np
from skimage.filters as rank

count = 0
x_tmp = x_train.copy()
y_tmp = y_train.copy()
print(len(x_train))
for img,label in zip(x_tmp,y_tmp):
    aug = [img]
    normal_result = rank.mean 


# In[ ]:




