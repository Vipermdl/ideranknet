#encoding:utf-8
import torch.utils.data as data
from utils.config import config
import numpy as np
import os.path as osp
import pandas as pd
import torch
import cv2

cfg = config


class Dataset(data.Dataset):
    def __init__(self, root_path, image_infos ,tramsform = None):
        self.root_path = root_path
        self.transform = tramsform
        self.image_infos = image_infos

    def __getitem__(self, index):
        
        #import pdb
        #pdb.set_trace()
        
        result = self.pull_item(index)
        return result

    def __len__(self):
        return len(self.image_infos)

    def pull_item(self,index):
        
        name = self.image_infos[index]['name']
        img = cv2.imread(osp.join(self.root_path,'JPEGImages/'+name+'.jpg'))
        label = self.image_infos[index]['label']
        if self.transform is not None:
            img, label = self.transform(img, label)
        return torch.from_numpy(img).permute(2,0,1), label

def convert_data(path,file_name):
    img_infos = list()
    data = pd.read_csv(osp.join(path, file_name))
    samples = data.iterrows()
    for i in range(data.count()['name']):
        row = next(samples)[1]
        label = row['label']
        name = row['name']
        sample = {'name': name, 'label': label}
        oversamples(img_infos, sample, 1)
    return img_infos

def convert1_data(path,file_name):
    img_infos = list()
    data = pd.read_csv(osp.join(path, file_name))
    samples = data.iterrows()
    for i in range(data.count()['name']):
        row = next(samples)[1]
        label = row['label']
        if label > 5:
            continue
        name = row['name']
        sample = {'name': name, 'label': label}
        oversamples(img_infos, sample, 1)
    return img_infos

def convert_rebalance_data(path,file_name):
    img_infos = list()
    data = pd.read_csv(osp.join(path, file_name))
    samples = data.iterrows()
    for i in range(data.count()['name']):
        row = next(samples)[1]
        label = row['label']
        name = row['name']
        sample = {'name': name, 'label': label}
        if label <= 3:
            oversamples(img_infos, sample, 2)#2200
        if label >3 and label < 3.5:
            #if np.random.randint(2) > 0:
            oversamples(img_infos, sample, 1)#3500
        if label >=3.5 and label < 4:
            #if np.random.randint(2) > 0:
            oversamples(img_infos, sample, 5)#3500      
        if label >= 4 and label < 5:
            oversamples(img_infos, sample, 15)#3300
        if label >= 5 and label < 6:
            oversamples(img_infos, sample, 35)#2500
        if label >= 6 and label < 7:
            oversamples(img_infos, sample, 100)#2700
        if label >= 7 and label < 8:
            oversamples(img_infos, sample, 436)#2900
        if label >= 8:
            oversamples(img_infos, sample, 120)#360

    return img_infos


def oversamples(infos,sample,sample_counts):
    count = 0
    while True:
        infos.append(sample)
        count += 1
        if count >= sample_counts:
            break

def get_loader(dataset, **kwargs):
    loader =data.DataLoader(dataset, cfg['batch_size'],
                          num_workers=cfg['num_workers'],
                          shuffle=True, drop_last=False,
                          pin_memory=True)#
    return loader


# ,collate_fn=detection_collate
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets





