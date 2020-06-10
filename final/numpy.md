# Numpy

                                ————Prepared by Ella Luo
                               



```python
import numpy as np  

a = np.array([1, 2, 3])  #a是向量： ndarray: 3个元素的向量

A = np.matrix([1, 2, 3]) #A 是1*3矩阵

print(a.shape)

print(A.shape)

print(type(A))
```

    (3,)
    (1, 3)
    <class 'numpy.matrix'>
    


```python
a = np.array([1, 2, 3])  
type(a)
```




    numpy.ndarray



实际上我们通常使用的是numpy array的表达方法来表达一个矩阵


```python
b = np.array([[1, 2, 3],[2, 3, 4]])   

print(b)

print(b.shape)
```

    [[1 2 3]
     [2 3 4]]
    (2, 3)
    

## numpy array的运算
注意区分\*和dot


```python
c = np.array([1, 2, 3]) 
d = np.array([4, 5, 6]) 

print(c*d) #对应位置相乘
print(np.multiply(c,d))
```

    [ 4 10 18]
    [ 4 10 18]
    


```python
a = np.array([1, 2, 3]) 
b = np.array([4, 5, 6]) 

print(np.dot(a,b))  #np.dot: 点积
print(a.dot(b))  
```

    32
    32
    


```python
x = np.array([[1, 2, 3],[2, 3, 4]])
y = np.array([[1, 2],[1, 3],[3, 4]]) 

print(x.dot(y))  
print(y.dot(x))

```

    [[12 20]
     [17 29]]
    [[ 5  8 11]
     [ 7 11 15]
     [11 18 25]]
    


```python
np.dot(x,y)
```




    array([[12, 20],
           [17, 29]])




```python
np.dot(y,x)
```




    array([[ 5,  8, 11],
           [ 7, 11, 15],
           [11, 18, 25]])




```python
A = np.matrix([[1, 2, 3],[2, 3, 4]])
B = np.matrix([[1, 2],[1, 3],[3, 4]])
print(A*B)
```

    [[12 20]
     [17 29]]
    

### 矩阵的转置
无论是numpy array的形式，还是numpy matrix的形式，都可以用 .T 或者 .transpose()来做转置运算


```python
A = np.matrix([1, 2, 3])
z = np.array([[1, 2, 3],[2, 3, 4]])

print(A.T)
print(A.transpose())

print(z.T)
print(z.transpose())
```

    [[1]
     [2]
     [3]]
    [[1]
     [2]
     [3]]
    [[1 2]
     [2 3]
     [3 4]]
    [[1 2]
     [2 3]
     [3 4]]
    


```python
A = np.matrix([1, 2, 3])
A.T   #A是1*3矩阵  A.T 是3*1矩阵
A.transpose()
```




    matrix([[1],
            [2],
            [3]])




```python
z = np.array([[1, 2, 3],[2, 3, 4]])  #z是2*3矩阵， z.T 是3*2矩阵
print(z.shape)
print(z.T)
print(z.transpose())
```

    (2, 3)
    [[1 2]
     [2 3]
     [3 4]]
    [[1 2]
     [2 3]
     [3 4]]
    

### 数字与numpy array的运算
无论是乘除还是加减，均是对该numpy array的每一项进行运算


```python
a = np.array([[1, 2, 3],[2, 3, 4]]) #2*3矩阵

print(3*a) #对矩阵里面每个数字都乘以3
print(a-1) #对矩阵里面每个数字都减去1
```

    [[ 3  6  9]
     [ 6  9 12]]
    [[0 1 2]
     [1 2 3]]
    

### numpy array的拼接和numpy matrix的拼接


```python
a = np.array([[1, 2, 3],[2, 3, 4]])
b = np.array([[6, 7, 8],[9, 10, 11]])

print(np.vstack((a,b))) #纵向拼接
print(np.hstack((a,b))) #horizontal stack  横向拼接

```

    [[ 1  2  3]
     [ 2  3  4]
     [ 6  7  8]
     [ 9 10 11]]
    [[ 1  2  3  6  7  8]
     [ 2  3  4  9 10 11]]
    

### 两种获取shortcuts的方法
1.linspace


```python
lin_spaced = np.linspace(0, 100, 21)   
#0是起点，100是终点， 21需要的数字的个数， 0和100 都会取到
#等间隔选取
print(lin_spaced)
```

    [  0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.  60.  65.
      70.  75.  80.  85.  90.  95. 100.]
    

2.arange


```python
aranged = np.arange(0, 100, 5)  

#0:起点(可以取到) 100：终点（终点取不到）， 5： 步长
print(aranged)


range1=np.arange(0,10,1)
print(range1)
```

    [ 0  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95]
    [0 1 2 3 4 5 6 7 8 9]
    

## reshape： 
在实际操作中，我们可能希望把一个1\*3的矩阵，或者3维的向量，变成3\*1的矩阵或者向量


```python
a = np.array([1,2,3])   #(3,) ndarray
a_reshape = a.reshape((3,1))   

print(a)   
print(a_reshape) 
```

    [1 2 3]
    [[1]
     [2]
     [3]]
    


```python
b=np.arange(0,10,1)
b_reshape=b.reshape(-1,1) #column=1,rows 无所谓

print(b)
print(b_reshape)
```

    [0 1 2 3 4 5 6 7 8 9]
    [[0]
     [1]
     [2]
     [3]
     [4]
     [5]
     [6]
     [7]
     [8]
     [9]]
    

## 已经学过的三种表示一串数的对比


```python
list_a = [1,2,3,4]  
list_a.append(5) 
print(list_a)
```

    [1, 2, 3, 4, 5]
    


```python
np_array_a = np.array([1,2,3,4]) 
```


```python
import pandas as pd
df = pd.DataFrame({'ella': [1, 2], 'cool': [3, 4]})
print(df)

```

       ella  cool
    0     1     3
    1     2     4
    


```python

```
