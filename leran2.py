#!/usr/bin/env python
# coding: utf-8

# # Day3 exercise

# ## 度（degree）をラジアン（radian）に変換する関数toradian()を作成せよ．

# In[1]:


import numpy as np

degree = float(input("Input degrees: "))
radian = degree*(np.pi/180.0)
print(radian)


# In[2]:


import numpy as np

def toradian(degree):
     return degree*(np.pi/180.0)

degree = 90.0
radian = toradian(degree)
print(radian)


# ##  反対に，ラジアン（radian）を度（degree）に変換する関数todegree()を作成せよ．

# In[3]:


import numpy as np

radian = float(input("Input radians: "))
degree = radian*(180.0/np.pi)
print(degree)


# In[4]:


import numpy as np

def todegree(radian):
    return radian*(180.0/np.pi)

radian = np.pi/2.0
degree = todegree(radian)
print(degree)


# ## 半径 r ，角度θから弧の長さlを求める関数arclength()を作成せよ．

# In[5]:


import numpy as np

def arclength(radius, angle):
    arc_length = (np.pi*2.0*radius) * (angle/360.0)
    return arc_length

radius = float(input('radius of circle: '))
angle = float(input('angle measure: '))
arc_length = arclength(radius, angle)
print("Arc Length is: ", arc_length)


# In[6]:


import numpy as np

def arclength(radius, angle):
    return  (np.pi*2.0*radius) * (angle/360.0)

radius = 1.0
angle = 180.0
arc_length = arclength(radius, angle)
print("Arc Length: ", arc_length)


# ## 半径r，角度θから扇型の面積 aを求める関数sectorarea()を作成せよ．

# In[7]:


import numpy as np

def sectorarea(radius , angle):
    sur_area = (np.pi*radius**2) * (angle/360.0)
    return sur_area

radius = float(input('radius of circle: '))
angle = float(input('angle measure: '))
sur_area = sectorarea(radius , angle)
print("Sector Area: ", sur_area)


# In[8]:


import numpy as np

def sectorarea(radius , angle):
    return (np.pi*radius**2) * (angle/360.0)

radius = 1.0
angle = 180.0
sur_area = sectorarea(radius , angle)
print("Sector Area: ", sur_area)


# ## 与えられたリスト L=[l1, l2, …, ln] の全要素のs = Σ liを計算する関数 list_sum()を作成せよ．

# In[9]:


def list_sum(l):
    sum = 0.0
    for i in l:
        sum += i
    return sum

sample = [1, 2, 3]
s = list_sum(sample)
print('list sum: ', s)


# In[10]:


def list_sum(l):
    sum = 0.0
    for i in l:
        sum += i
    return sum

sample = [1, 2, 3]
s = list_sum(sample)
print(s)
print(sum(sample))


# ## 3X3個の正方形の方陣に1から9までの数字を配置し，各行・各列・各対角の和がいずれも15になるには，どのように数字を配置すれば良いかをもとめよ．（対象形を除けば三方陣の解は1つである． [[8 1 6], [3 5 7], [4 9 2]]）

# In[11]:


# [Pythonで順列・組み合わせを求める - Qiita](https://qiita.com/BlueSilverCat/items/77f4e11d3930d7b8959b)

# list(itertools.permutations(data, r))

def listExcludedIndices(data, indices=[]):
  return [x for i, x in enumerate(data) if i not in indices]

data = [1,2,3]
result = []
for i in range(len(data)):
  for j in range(len(data) - 1):
    for k in range(len(data) - 2):
      jData = listExcludedIndices(data, [i])
      kData = listExcludedIndices(jData, [j])
      result.append([data[i], jData[j], kData[k]])

print(result)


# In[12]:


import numpy as np

def perm(r_list):
    if (len(r_list) == 1):
        return r_list
    else:
        head = r_list[0]
        body = r_list[1:len(r_list)]
        print(head, body)
            
a = np.arange(10)
print(a)
perm(a)


# In[13]:


# 3x3 magic square: [8, 1, 6, 3, 5, 7, 4, 9, 2]
import itertools
import numpy as np

# 1から9までの数字の全順列を求める
per = itertools.permutations(range(1,10))

# 求めた全順列の各数列が魔方陣かどうかをチェックする
count = 0 #　見つけた魔方陣の個数
for i in per:
    flag = False # 和が15ではない時のフラグ（途中でチェックを省略するために使用する）
    j = np.array(i).reshape(3,3) # 1x9 -> 3x3に変形する

    # 各列の和をチェック
    if flag == False:
        for k in j.sum(axis=0):
            if 15 != k:
                flag = True
                break
                
    # 各行の和をチェック
    if flag == False:
        for k in j.sum(axis=1):
            if 15 != k:
                flag = True
                break
                
    # 対角の和をチェック
    if flag == False:
        if 15 != np.diag(j).sum():
                flag = True

    # 逆対角の和をチェック
    if flag == False:
        if 15 != np.diag(np.fliplr(j)).sum():
            flag = True

    # 魔方陣であればデータを出力する
    if flag == False:
        count += 1
        print(i)

print("# of magic square: " , count)


# In[14]:


import math
print(math.factorial(5))


# ## 自作の順列関数

# In[1]:


def permutation(lst):
    if type(lst) != list:
        print('not list')
        return [lst]
    
    if len(lst) == 1:
        return [lst]

    perm = []
    for i, x in enumerate(lst):
        rlst = lst.copy()
        del rlst[i]
        for j in permutation(rlst):
            j.insert(0, x)
            perm.append(j)

    return perm

a = list(range(1,10))
b = permutation(a)
print(b[:5])
print(len(b))


# In[16]:


import itertools
import numpy as np

# 1から9までの数字の全順列を求める
per = itertools.permutations(range(1,10))
print(len(list(per)))


# ## itertoolsを使用しない魔方陣

# In[17]:


# 3x3 magic square: [8, 1, 6, 3, 5, 7, 4, 9, 2]
import itertools
import numpy as np

# 1から9までの数字の全順列を求める
def permutation(lst):
    if type(lst) != list:
        print('not list')
        return [lst]
    
    if len(lst) == 1:
        return [lst]

    perm = []
    for i, x in enumerate(lst):
        rlst = lst.copy()
        del rlst[i]
        for j in permutation(rlst):
            j.insert(0, x)
            perm.append(j)

    return perm

per =permutation(list(range(1,10)))

# 求めた全順列の各数列が魔方陣かどうかをチェックする
count = 0 #　見つけた魔方陣の個数
for i in per:
    flag = False # 和が15ではない時のフラグ（途中でチェックを省略するために使用する）
    j = np.array(i).reshape(3,3) # 1x9 -> 3x3に変形する

    # 各列の和をチェック
    if flag == False:
        for k in j.sum(axis=0):
            if 15 != k:
                flag = True
                break
                
    # 各行の和をチェック
    if flag == False:
        for k in j.sum(axis=1):
            if 15 != k:
                flag = True
                break
                
    # 対角の和をチェック
    if flag == False:
        if 15 != np.diag(j).sum():
                flag = True

    # 逆対角の和をチェック
    if flag == False:
        if 15 != np.diag(np.fliplr(j)).sum():
            flag = True

    # 魔方陣であればデータを出力する
    if flag == False:
        count += 1
        print(i)

print("# of magic square: " , count)


# In[ ]:




