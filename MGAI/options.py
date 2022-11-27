import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=200,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-4,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=10,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=240,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=50,    help='every n epochs decay learning rate')
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

parser.add_argument('--rgb_label_root',  type=str, default='/home/lh830/huanglm/dataset/RGBT/vt5000/Train/RGB//', help='the training rgb images root')
parser.add_argument('--t_label_root',    type=str, default='/home/lh830/huanglm/dataset/RGBT/vt5000/Train/T//', help='the training t/depth images root')
parser.add_argument('--gt_label_root',   type=str, default='/home/lh830/huanglm/dataset/RGBT/vt5000/Train/GT//', help='the training gt images root')

parser.add_argument('--save_path',  type=str, default='./Checkpoint/MGAI/', help='the path to save models and logs')

opt = parser.parse_args()

