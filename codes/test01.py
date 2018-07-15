#encoding:utf-8
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.nn as nn
from utils.visualize import *
from utils.utils import progress_bar
from utils.augmentations import *
import torch.optim as optim
import torch.nn.init as init
import time
import argparse
from getData.dataset import *
from models import *
from utils.config import config
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(1)

cfg = config
start_epoch = 0
best_acc = float('inf')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    #if isinstance(m, nn.Conv2d):
        #xavier(m.weight.data)
        #m.bias.data.zero_()
    if isinstance(m, nn.Linear):
        xavier(m.weight.data)
        m.bias.data.zero_()


parser = argparse.ArgumentParser(
    description='the regression for image diffculty with MobileNet'
)
parser.add_argument('--dataset_root',default='/home/kawayi-2/mdl/VOCdevkit/VOC2012', help='Dataset root directory path')
parser.add_argument('--save_folder', default='checkpoint/', help='Directory for saving checkpoint models')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom for loss visualization')


args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    

# load the data
train_img_infos = convert_rebalance_data('./getData','train_dataset.csv')
test_img_infos = convert_data('./getData', 'test_dataset.csv')
val_img_infos = convert_data('./getData', 'val_dataset.csv')

train_dataset = Dataset(root_path=args.dataset_root, image_infos=train_img_infos, tramsform=Augmentation2train())
val_dataset = Dataset(root_path=args.dataset_root, image_infos=val_img_infos, tramsform=Augmentation2test())
test_dataset = Dataset(root_path=args.dataset_root, image_infos=test_img_infos, tramsform=Augmentation2test())

train_loader = get_loader(train_dataset)
val_loader = get_loader(val_dataset)
test_loader = get_loader(test_dataset)

# Model
print('==> Building model..')
# net = VGG('VGG19')
#dense_net = ResNet34()


dense_net = myResNet101()

resnet101 = models.resnet101(pretrained=True)
 
pretrained_dict = resnet101.state_dict()  

model_dict = dense_net.state_dict()  
 
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  

model_dict.update(pretrained_dict)  
 
dense_net.load_state_dict(model_dict)  
  

net = dense_net


# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()

# dense_net = MobileNet()
# net = dense_net

# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

#net.apply(weights_init)


if args.cuda:
    net = torch.nn.DataParallel(dense_net,[0,1,2])
    cudnn.benchmark = True


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

vis = Visualizer(env='img_diff')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    groups = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.float().cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
    
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += (loss.data[0])
        
        groups += 1
    vis.plot('train_loss',train_loss / groups)
    progress_bar(epoch, len(train_loader), 'Loss: %.3f' % (train_loss/groups))

def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0.0
    groups = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):

        if args.cuda:
            inputs = Variable(inputs.cuda(), volatile = True)
            targets = Variable(targets.float().cuda())

        outputs = net(inputs)
        
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        groups += 1

    epoch_loss = val_loss / groups
    print('\nThe epoches of {} Val loss is {}'.format(epoch,epoch_loss))
    vis.plot('test_loss',epoch_loss)
    # Save checkpoint.
    if epoch_loss < best_acc:
        print('Saving..')
        torch.save(net.state_dict(), './checkpoint/net_resnet101.pth')
        best_acc = epoch_loss


def test():
    #would to load the best model in val process
    dense_net = myResNet101()
    net = dense_net
    if args.cuda:
      net = torch.nn.DataParallel(dense_net)
      cudnn.benchmark = True
    net.load_state_dict(torch.load('./checkpoint/net_resnet.pth'))

    net.eval()
    test_loss = 0.0
    groups = 0
    predict = np.array(0.0)
    target = np.array(0.0)
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(test_loader):

        if args.cuda:
            inputs = Variable(inputs.cuda(), volatile = True)
            targets = Variable(targets.float().cuda())

        outputs = net(inputs)
        
        predict = np.vstack((predict,outputs.data.cpu().numpy()))
        target = np.vstack((target,targets.view(-1,1).data.cpu().numpy()))
        
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        groups += 1

    epoch_loss = test_loss / groups
    print('\nThe image test time is'.format((time.time()-start_time)/len(test_loader)))
    return epoch_loss, predict[1:], target[1:]
    

def measure_kendall(predict, target):
    num = 0
    for i in range(1, len(predict)): #1 --  n-1
        for j in range(i):
            num += np.sign(predict[i]-predict[j]) * np.sign(target[i]-target[j])
    return 2 * num / (len(predict) * (len(predict)-1))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = cfg['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
#    for epoch in range(start_epoch, start_epoch+280):
#       if epoch in cfg['lr_steps']:
#          adjust_learning_rate(optimizer, cfg['gamma'], epoch)
#       train(epoch)
#       val(epoch)
#       print(best_acc)

    loss, predict, target = test()
    list_predict = list()
    list_target = list()
    for i in range(len(predict)):
        list_predict.append(predict[i][0])
        list_target.append(target[i][0])

    #draw the picture
    plt.scatter(list_predict,list_target,color='blue')
    kendall  = measure_kendall(list_predict, list_target)
    print('\nThe test dataset loss is {}'.format(loss))
    print('\nThe test dataset kendall is {}'.format(kendall))
    plt.savefig("./predict_target.jpg")