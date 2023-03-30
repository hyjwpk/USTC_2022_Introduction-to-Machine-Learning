import pandas as pd
import numpy as np


def ent(data):
    prob = pd.value_counts(data)/len(data)
    return sum(np.log2(prob)*prob*(-1))


def get_info_gain(data, feat, label):
    if feat == 'C':
        for i in range(7):
            _data = data.copy()
            _data[feat] = _data[feat].apply(lambda x: x>i+1.5)
            e1 = _data.groupby(feat).apply(lambda x: ent(x[label]))
            p1 = pd.value_counts(_data[feat])/len(_data[feat])
            e2 = sum(e1*p1)
            print(round((ent(_data[label]) - e2),3))
    else:
        e1 = data.groupby(feat).apply(lambda x: ent(x[label]))
        p1 = pd.value_counts(data[feat])/len(data[feat])
        e2 = sum(e1*p1)
        print(round((ent(data[label]) - e2),3))

def get_gain_ratio(data,feat,label):
    if feat == 'C':
        for i in range(7):
            _data = data.copy()
            _data[feat] = _data[feat].apply(lambda x: x>i+1.5)
            e1 = _data.groupby(feat).apply(lambda x: ent(x[label]))
            p1 = pd.value_counts(_data[feat])/len(_data[feat])
            e2 = sum(e1*p1)
            iv = -sum(p1*np.log2(p1))
            if iv==0:
                print(0)
            else:
                print(round((ent(_data[label]) - e2)/iv,3))
    else:
        e1 = data.groupby(feat).apply(lambda x: ent(x[label]))
        p1 = pd.value_counts(data[feat])/len(data[feat])
        e2 = sum(e1*p1)
        iv = -sum(p1*np.log2(p1))
        if iv==0:
            print(0)
        else:
            print(round((ent(data[label]) - e2)/iv,3))



data = pd.DataFrame({'A': ['T','T','T','F','F','F','F','T','F','F'], 
                     'B': ['T','T','F','F','T','T','F','F','T','F'],
                     'C': [1.0,6.0,5.0,4.0,7.0,3.0,8.0,7.0,5.0,2.0],
                     '类别': ['是', '是', '否', '是', '否', '否', '否', '是', '否', '是']})
data = data.iloc[[1,2,7],:]
print(data.head())
label = '类别'
print('信息增益')
for feat in ['A', 'B', 'C']:
    get_info_gain(data, feat, label)

print('增益率')
for feat in ['A', 'B', 'C']:
    get_gain_ratio(data, feat, label)
