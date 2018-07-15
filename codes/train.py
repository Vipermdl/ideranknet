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
from addMoreData.kendallLoss import * 
from utils.config import config
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import kl_div
import os

torch.manual_seed(1)

cfg = config
start_epoch = 0
best_acc = float('inf')
best_kendall = 0



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
parser.add_argument('--cuda', default=False, type=str2bool, help='Use CUDA to train model')
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
# train_img_infos = convert_rebalance_data('./getData','train_dataset.csv')
train_img_infos = convert_data('./getData','train_dataset.csv')
test_img_infos = convert1_data('./getData', 'test_dataset.csv')
val_img_infos = convert_data('./getData', 'val_dataset.csv')

print(len(test_img_infos))
exit(0)

train_dataset = Dataset(root_path=args.dataset_root, image_infos=train_img_infos, tramsform=Augmentation2train())
val_dataset = Dataset(root_path=args.dataset_root, image_infos=val_img_infos, tramsform=Augmentation2test())
test_dataset = Dataset(root_path=args.dataset_root, image_infos=test_img_infos, tramsform=Augmentation2test())

train_loader = get_loader(train_dataset)
val_loader = get_loader(val_dataset)
test_loader = get_loader(test_dataset)

# Model
print('==> Building model..')

dense_net = myResNet152()

resnet152 = models.resnet152(pretrained=True)
 
pretrained_dict = resnet152.state_dict()  

model_dict = dense_net.state_dict()  
 
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  

model_dict.update(pretrained_dict)  
 
dense_net.load_state_dict(model_dict)  
  

net = dense_net

#net.apply(weights_init)


if args.cuda:
    net = torch.nn.DataParallel(dense_net,[0,1,2,3])
    cudnn.benchmark = True


criterion_mse = nn.MSELoss()
criterion_kendall = kendallloss()

optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

vis = Visualizer(env='img_diff')

# Training
def train(epoch, lamba):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_kendall_loss = 0.0
    train_mse_loss = 0.0
    train_loss = 0.0
    groups = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.float().cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        
#        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
#        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            
        optimizer.zero_grad()
        outputs = net(inputs)
        
        #kl = kl_div(outputs, targets)
        #vis.plot('train_KL_epoch',kl.data[0])
#        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
 
        loss_mse = criterion_mse(outputs,targets)
        loss_kendall = criterion_kendall(outputs,targets)

        loss = lamba * loss_kendall + (1 - lamba) * loss_mse               
                
        loss.backward()
        optimizer.step()
        train_loss += (loss.data[0])
        train_mse_loss += (loss_mse.data[0])
        train_kendall_loss += (loss_kendall.data[0])
        
        groups += 1
    vis.plot('train_loss',train_loss / groups)
    vis.plot('train_mse_loss',train_mse_loss / groups)
    vis.plot('train_kendall_loss',train_kendall_loss / groups)
    #progress_bar(epoch, len(train_loader), 'Loss: %.3f' % (train_loss/groups))

def val(epoch, lamba):
    global best_acc
    global best_kendall
    
    net.eval()
    val_loss = 0.0
    val_kendall_loss = 0.0
    val_mse_loss = 0.0
    
    groups = 0
    predict = np.array(0.0)
    target = np.array(0.0)

    for batch_idx, (inputs, targets) in enumerate(val_loader):

        if args.cuda:
            inputs = Variable(inputs.cuda(), volatile = True)
            targets = Variable(targets.float().cuda())

        outputs = net(inputs)
        #kl = kl_div(outputs, targets)
        #vis.plot('val_KL_epoch',kl.data[0])
        
        predict = np.vstack((predict,outputs.data.cpu().numpy()))[1:]
        target = np.vstack((target,targets.view(-1,1).data.cpu().numpy()))[1:]
        
        loss_mse = criterion_mse(outputs,targets)
        loss_kendall = criterion_kendall(outputs,targets)

        loss = lamba * loss_kendall + (1-lamba) * loss_mse       

        val_loss += loss.data[0]
        val_kendall_loss += loss_kendall.data[0]
        val_mse_loss += loss_mse.data[0]
        
        groups += 1

    
    #print('\nThe epoches of {} Val loss is {}'.format(epoch,epoch_loss))
    vis.plot('val_mse_loss',val_mse_loss / groups)
    vis.plot('val_kendall_loss',val_kendall_loss / groups)
    vis.plot('val_loss',val_loss / groups)
        
    list_predict = list()
    list_target = list()
    for i in range(len(predict)):
        list_predict.append(predict[i][0])
        list_target.append(target[i][0])
    kendall  = measure_kendall(list_predict, list_target)
    vis.plot('val kendall in practice',kendall)
    
    epoch_loss = val_mse_loss / groups  
    # Save checkpoint.
    if kendall > best_kendall and epoch_loss <= 0.235:
        print('Saving..')
        torch.save(net.state_dict(), './checkpoint/net_test.pth')
        #best_acc = epoch_loss
        best_kendall = kendall

def test():
    #would to load the best model in val process
    dense_net = myResNet152()
    net = dense_net
    if args.cuda:
      net = torch.nn.DataParallel(dense_net)
      cudnn.benchmark = True
    net.load_state_dict(torch.load('./checkpoint/sota.pth'))

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
        
        loss = criterion_mse(outputs,targets)
        test_loss += loss.data[0]
        groups += 1

    epoch_loss = test_loss / groups
    #print('\nThe image test time is'.format((time.time()-start_time)/len(test_loader)))
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

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        #lam = np.random.beta(alpha, alpha)
        lam = 0.5
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
    #lamba = 0.3
    #while lamba < 1:
#    for epoch in range(start_epoch, start_epoch+210):
#       if epoch % 3 == 0 :
#           lamba = 0
#       else:
#           lamba = 1
#       if epoch in cfg['lr_steps']:
#          adjust_learning_rate(optimizer, cfg['gamma'], epoch)
#       train(epoch, lamba)
#       val(epoch, lamba)

    loss, predict, target = test()
    list_predict = list()
    list_target = list()
    x_lim = [2.5,3,4,5,6]
    for i in range(len(predict)):
        #if target[i][0] > 5:
        list_predict.append(predict[i][0])
        list_target.append(target[i][0])
 
    kendall  = measure_kendall(list_predict, list_target)
    
    import pickle
    predict_output = open('predict.pkl', 'wb')
    pickle.dump(list_predict,predict_output)
    
    target_output = open('target.pkl', 'wb')
    pickle.dump(list_target,target_output)
    target_output.close()
    predict_output.close()

    #print(lamba)
    print('\nThe test dataset loss is {}'.format(loss))
    print('\nThe test dataset kendall is {}'.format(kendall))
    #lamba += 0.1