#encoding:utf-8
import pandas as pd
import numpy as np
import os.path as osp
import cv2


totalpd = pd.read_csv('../getData/train_dataset.csv')
name2list = totalpd['name'].tolist()
score2list = totalpd['label'].tolist()

# img  name  score
def addData(pd, time, img_path, toname, alpha = 1.0):
    name = pd['name'].tolist()
    score = pd['label'].tolist()
    for i in range(time):
        index_a = np.random.randint(0,len(name))
        index_b = np.random.randint(0, len(name))
        img_a = cv2.imread(osp.join(img_path, 'JPEGImages/' + name[index_a] + '.jpg'))
        img_b = cv2.imread(osp.join(img_path, 'JPEGImages/' + name[index_b] + '.jpg'))
        img_a = cv2.resize(img_a, (224, 224))
        img_b = cv2.resize(img_b, (224, 224))
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * img_a + (1 - lam) * img_b
        mixed_y = lam * score[index_a] + (1 - lam) * score[index_b]
        name2list.append(toname+str(i))
        score2list.append(mixed_y)
        cv2.imwrite(osp.join(img_path, 'JPEGImages/' + toname+str(i) + '.jpg'), mixed_x)



pd_3 = totalpd[totalpd['label'] < 3]
pd_4 = totalpd[totalpd['label'] < 4]
pd_5 = totalpd[totalpd['label'] < 5]
pd_6 = totalpd[totalpd['label'] < 6]
pd_7 = totalpd[totalpd['label'] < 7]
pd_8 = totalpd[totalpd['label'] < 8]

pd_3 = totalpd[totalpd['label'] < 3]
pd_3_4 = pd_4[pd_4['label'] > 3]
pd_4_5 = pd_5[pd_5['label'] > 4]
pd_5_6 = pd_6[pd_6['label'] > 5]
pd_6_7 = pd_7[pd_7['label'] > 6]
pd_7_8 = pd_8[pd_8['label'] > 6]

img_path = '/home/kawayi-2/mdl/VOCdevkit/VOC2012'
# addData(pd_3, 2000, img_path, 'syn2to3')
# addData(pd_3_4, 2000, img_path, 'syn3to4')
addData(pd_4_5, 2800, img_path, 'syn4to5')
addData(pd_5_6, 2000, img_path, 'syn5to6')
addData(pd_6_7, 1000, img_path, 'syn6to7')
addData(pd_7_8, 500, img_path, 'syn7to8')


df_train = pd.DataFrame({'name': name2list, 'label': score2list})
df_train.to_csv('../getData/fake_train_dataset.csv',columns=['name','label'])


