#encoding:utf-8
from getData.dataset import convert_data
import pandas as pd
import numpy as np
import os.path as osp

np.random.seed(19)

root_path = './'

def split_data(root_path):
    train_imgs_name = list()
    test_imgs_name = list()
    val_imgs_name = list()
    train_imgs_score = list()
    test_imgs_score = list()
    val_imgs_score = list()

    data = pd.read_csv(osp.join(root_path, 'VSD_dataset.csv'))
    samples = data.iterrows()
    for i in range(data.count()['name']):
        row = next(samples)[1]
        label = row['score']
        name = row['name']
        seed = np.random.randint(0, 8)
        if  seed < 2:
            test_imgs_name.append(name)
            test_imgs_score.append(label)
        elif seed == 2 or seed == 3:
            val_imgs_name.append(name)
            val_imgs_score.append(label)
        else:
            train_imgs_name.append(name)
            train_imgs_score.append(label)
    df_train = pd.DataFrame({'name':train_imgs_name,'label':train_imgs_score})
    df_test = pd.DataFrame({'name': test_imgs_name, 'label': test_imgs_score})
    df_val = pd.DataFrame({'name': val_imgs_name, 'label': val_imgs_score})

    df_train.to_csv('train_dataset.csv',columns=['name','label'])
    df_test.to_csv('test_dataset.csv',columns=['name','label'])
    df_val.to_csv('val_dataset.csv',columns=['name','label'])

if __name__ == '__main__':
    root_path = './'
    # split_data(root_path)
    # train = convert_data('./','train_dataset.csv')
    # print(len(train))
    # test = convert_data('./', 'test_dataset.csv')
    # print(len(test))
    # val = convert_data('./', 'val_dataset.csv')
    # print(len(val))
    train_set = pd.read_csv('train_dataset.csv')
    print()

