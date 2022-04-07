#!/usr/bin/env python
# coding: utf-8

import numpy as np


# ## Numpy.arange()を使用し，10から27までの数列z1を作成
z1 = np.arange(10,28)
print(z1)
print(len(z1))


# ## 上記で作成した数列z1を反転（順序を逆にする）した数列z2を作成
z2 = z1[::-1]
print(z2)
print(len(z2))


# ## 上記で作成した数列z2を前半（0番目から8番目）と後半（9番目から17番目）で分けた数列z21とz22を作成せよ．
z21 = z2[:len(z2)//2]
z22 =  z2[len(z2)//2:]
print(z21, z22)
print(len(z21), len(z22))


# ## 上記で作成した数列z21を3X3（3行3列）のreshape()を使用し行列z31に変換せよ．同様に，数列z22から行列z32を作成せよ．
z31 = z21.reshape(3,3)
z32 = z22.reshape(3,3)
print(z31)
print(z32)


# ## 行列z31の固有値wと固有ベクトルvを求めよ．
w, v =  np.linalg.eig(z31)
print('固有値: ', w)
print('固有ベクトル: ', v)


# ## 固有値wから最初の固有値w0を取り出せ．

# In[7]:


print(w[0])


# ## 固有値w0の固有ベクトルv0を固有ベクトルｖから取り出せ．

# In[8]:


print(v[:,0])


# ## 上記で求めた固有値w0と固有ベクトルｖ0を使用し，$（z_{31} - w_0I）\vec{v_0} \approx 0$ を確認せよ．$I$ は単位行列である．

# In[9]:


i = np.eye(3)
print((z31 - w[0] * i)@v[:,0])


# In[10]:


v[0]


# In[11]:


a = np.arange(1,10)
b = a
print(a*b)


# ## 自身が自分自身を除く正の約数の和に等しくなる正の整数を完全数（perfect number）と呼ぶ．例えば，最初の完全数6は「6=1+2+3」，次の完全数28は「28=1+2+4+7+14」である．この完全数を求める関数 ``perfect_number()`` を作成せよ．作成した関数を使用し，10000以下の全ての完全数を求めよ．(必要なければNumpyを使用しなくても問題ありません．)

# In[12]:


# 全ての約数をリストで返す関数
import numpy as np

def divisor(n):
    if n == 1:
        return [1]

    d = [1, n]
    for i in range(2, int(np.floor(np.sqrt(n)))+1):
        if n % i == 0:
            d.append(i)
            if i != n//i:
                d.append(n//i)

    return sorted(d)

# 完全数をもとめる
n = 10000
for i in range(1,n+1):
    div_list = divisor(i)
    # 自分自身を削除する
    div_list.pop()
    if i == np.sum(div_list):
        print(i)


# In[ ]:
