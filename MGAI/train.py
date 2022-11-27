import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from MGAI import MGAINet
from Code.utils.data import get_loader,test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt



#set the device for training
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')

  
cudnn.benchmark = True

#build the model
model = MGAINet()
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)
    
model.cuda()
params    = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


#set the path
train_image_root = opt.rgb_label_root
train_gt_root    = opt.gt_label_root
train_t_root = opt.t_label_root


save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root,train_t_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step   = len(train_loader)


logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("MGAI_unif-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

#set loss function
CE   = torch.nn.BCEWithLogitsLoss()

step = 0
writer     = SummaryWriter(save_path+'summary')
best_mae   = 1
best_epoch = 0

print(len(train_loader))

def bce_iou_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    bce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return (bce+iou).mean()

def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, ts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images = images.cuda()
            gts    = gts.cuda()
            ts     = ts.cuda()

            pre_res  = model(images,ts)
            
            loss1    = bce_iou_loss(pre_res[0], gts)
            loss2    = bce_iou_loss(pre_res[1], gts)
            loss3    = bce_iou_loss(pre_res[2], gts)

            loss_seg = loss1 + loss2 + loss3

            loss = loss_seg 
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 50 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data,  loss3.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'MGAINet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'MGAINet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_loader, model, optimizer, epoch,save_path)

