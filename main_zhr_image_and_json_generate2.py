#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.init as init
from model.ctrgcn_zhr import Temporal_CNN_Model
from model.loss import CustomMSELoss

from torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            # self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.Model_zhr_temp = Temporal_CNN_Model()
        print('这一步没问题1')
        # simility_array_17 = np.load('data/ntu/旋转的自相似矩阵_17.npy').reshape(64,64)
        tensor = torch.tensor(
       [[[[0.8, 0.7, 0.6, 0.5, 0.9],
          [0.7, 0.8, 0.7, 0.6, 0.9],
          [0.6, 0.7, 0.8, 0.7, 0.9],
          [0.5, 0.6, 0.7, 0.8, 0.9],
          [0.4, 0.5, 0.6, 0.7, 0.9]]]])
        print(tensor)
        weights = torch.load('work_dir_train/runs-zhr_try_to_train_fourth_3.pt')
        self.Model_zhr_temp.load_state_dict(weights)
        print('这一步没问题2')
        self.Model_zhr_temp(tensor)
        # numpy_array = self.Model_zhr_temp(torch.from_numpy(simility_array_17).unsqueeze(0).unsqueeze(1).to(torch.float32)).detach().numpy()
        # np.save('data/ntu/array_17_after_cnn.npy', numpy_array)
        print('这一步没问题3')
        state_dict = self.Model_zhr_temp.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        # print(self.arg.model_saved_name)
        # torch.save(weights, self.arg.model_saved_name + '-' + 'zhr_first' + '.pt')
        print('这一步没问题4')
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.loss_temporary = CustomMSELoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
        self.optimizer_zhr = optim.SGD(
                self.Model_zhr_temp.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
 
        count_file_all = 0
        for file in os.listdir('./data/ntu/train_npy_csv标准格式'):
            # print(file[:-9])
            print(count_file_all)
            count_file_all += 1

            final_numpy = np.empty((0, 6400))
            kkk = np.load('data/ntu/train_npy_csv标准格式/{}'.format(file))
            kkk = torch.from_numpy(kkk).unsqueeze(0)

            random_number = random.randint(0, 24)
            
            for i in range(int(kkk.shape[2]/64)):
                k = kkk[:, :, i*64:(i+1)*64, :, :]
                k = k.float().cuda(self.output_device)
                k[:, 2, :, :, :] = 0

                k[:, :, :, random_number, :] = 0
                k[:, :, :, (random_number+1)%25, :] = 0
                k[:, :, :, (random_number+2)%25, :] = 0
                k[:, :, :, (random_number+3)%25, :] = 0
                k[:, :, :, (random_number+4)%25, :] = 0
                k[:, :, :, (random_number+5)%25, :] = 0
                k[:, :, :, (random_number+6)%25, :] = 0
                k[:, :, :, (random_number+7)%25, :] = 0
                k[:, :, :, (random_number+8)%25, :] = 0
                k[:, :, :, (random_number+9)%25, :] = 0

                # for j in range(16):
                #     x_tensor_1,output_1 = self.model(k[:, :, j*4:(j+1)*4, :, :], torch.tensor([1]).long().cuda(self.output_device), torch.tensor([1]).long().cuda(self.output_device))
                #     x_tensor_1 = x_tensor_1.mean(1).view(4,-1)
                #     x_tensor_1 = x_tensor_1.cpu().detach().numpy()
                #     final_numpy = np.vstack((final_numpy, x_tensor_1))
                x_tensor_1,output_1 = self.model(k, torch.tensor([1]).long().cuda(self.output_device), torch.tensor([1]).long().cuda(self.output_device))
                x_tensor_1 = x_tensor_1.mean(1).view(64,-1)
                x_tensor_1 = x_tensor_1.cpu().detach().numpy()
                final_numpy = np.vstack((final_numpy, x_tensor_1))
            # print(final_numpy.shape)
            # if int(kkk.shape[2]/64)*64 < kkk.shape[2]:
            #     k = kkk[:, :, int(kkk.shape[2]/64)*64:, :, :]
            #     k = k.float().cuda(self.output_device)
            #     k[:, 2, :, :, :] = 0
            #     x_tensor_1,output_1 = self.model(k, torch.tensor([1]).long().cuda(self.output_device), torch.tensor([1]).long().cuda(self.output_device))
            #     x_tensor_1 = x_tensor_1.mean(1).view(x_tensor_1.shape[0],-1)
            #     x_tensor_1 = x_tensor_1.cpu().detach().numpy()
            #     final_numpy = np.vstack((final_numpy, x_tensor_1))
            np.save('../../../../data/ssd1/zhanghaoran/zhr/pose_action_feature_遮挡十处/{}(用我的模型跑的,时间幅度为64).npy'.format(file[:-9]),final_numpy)


        # final_numpy = np.empty((0, 6400))
        # kkk = np.load('data/ntu/train_npy_csv标准格式/stu2_41_numpy的标准格式.npy')
        # kkk = torch.from_numpy(kkk).unsqueeze(0)
        # for i in range(int(kkk.shape[2]/64)):
        #     # print(i)
        #     k = kkk[:, :, i*64:(i+1)*64, :, :]
        #     k = k.float().cuda(self.output_device)
        #     k[:, 2, :, :, :] = 0
        #     # for j in range(16):
        #     #     x_tensor_1,output_1 = self.model(k[:, :, j*4:(j+1)*4, :, :], torch.tensor([1]).long().cuda(self.output_device), torch.tensor([1]).long().cuda(self.output_device))
        #     #     x_tensor_1 = x_tensor_1.mean(1).view(4,-1)
        #     #     x_tensor_1 = x_tensor_1.cpu().detach().numpy()
        #     #     final_numpy = np.vstack((final_numpy, x_tensor_1))
        #     x_tensor_1,output_1 = self.model(k, torch.tensor([1]).long().cuda(self.output_device), torch.tensor([1]).long().cuda(self.output_device))
        #     x_tensor_1 = x_tensor_1.mean(1).view(64,-1)
        #     x_tensor_1 = x_tensor_1.cpu().detach().numpy()
        #     final_numpy = np.vstack((final_numpy, x_tensor_1))
        # np.save('单个视频的64帧特征10/stu2_41_numpy(benchmark模型跑的,时间幅度为64).npy',final_numpy)

        return
        



        for batch_idx, (data_1, label_1, index, data_2, label_2, index_2) in enumerate(process):
            self.global_step += 1
            # if label_1.item()==5:
            #     np.save('data/ntu/提取成64帧的原始骨骼数据/64帧原始数据_{}.npy'.format(index.item()), data_1)
            #     np.save('data/ntu/提取成64帧的原始骨骼数据/64帧旋转数据_{}.npy'.format(index.item()), data_2)
            # continue
            print('data1:',data_1.shape,type(data_1))
            with torch.no_grad():
                data_1 = data_1.float().cuda(self.output_device)
                label_1 = label_1.long().cuda(self.output_device)
                data_2 = data_2.float().cuda(self.output_device)
                label_2 = label_2.long().cuda(self.output_device)                
            timer['dataloader'] += self.split_time()

            # forward
            # x_tensor_1 = torch.tensor([]).to(self.output_device)
            # output_1 = torch.tensor([]).to(self.output_device)
            # subtensor_shape = (1, 3, 4, 25, 2)
            # subtensors_1 = data_1.reshape(16, *subtensor_shape)
            # for i, sub in enumerate(subtensors_1):
            #     x1,out1 = self.model(sub, label_1[i*4:(i+1)*4], index[i*4:(i+1)*4])
            #     # print('看看out',type(out1),out1.shape)
            #     x_tensor_1 = torch.cat((x_tensor_1, x1))
            #     output_1 = torch.cat((output_1, out1))

            # x_tensor_2 = torch.tensor([]).to(self.output_device)
            # output_2 = torch.tensor([]).to(self.output_device)
            # subtensor_shape = (1, 3, 4, 25, 2)
            # subtensors_2 = data_2.reshape(16, *subtensor_shape)
            # for i, sub in enumerate(subtensors_2):
            #     x2,out2 = self.model(sub, label_2[i*4:(i+1)*4], index[i*4:(i+1)*4])
            #     # print('看看out',type(out2),out2.shape)
            #     x_tensor_2 = torch.cat((x_tensor_2, x2))
            #     output_2 = torch.cat((output_2, out1))
            data_1[:, 2, :, :, :] = 0
            data_2[:, 2, :, :, :] = 0
            x_tensor_1,output_1 = self.model(data_1, label_1, index)
            x_tensor_2,output_2 = self.model(data_2, label_2, index)



            similarity_numpy = np.array([]) 
            label_zhr = np.array([]) 
            x_tensor_1 = x_tensor_1.mean(1).view(64,-1)
            np.save('单个视频的64帧特征9/x_tensor_origin_{}.npy'.format(index.item()),x_tensor_1.cpu().detach().numpy())
            x_tensor_2 = x_tensor_2.mean(1).view(64,-1)
            np.save('单个视频的64帧特征9/x_tensor_rotate_{}.npy'.format(index.item()),x_tensor_2.cpu().detach().numpy())
            name_numpy = np.append(name_numpy,index.item())
            label_numpy = np.append(label_numpy,label_1.item())
            
            if batch_idx == 200:
                np.savez('单个视频的64帧特征/my_arrays.npz', name_numpy=name_numpy, label_numpy=label_numpy)
                break
            for i in range(x_tensor_1.size(0)):
                norm_1 = torch.norm(x_tensor_1[i])
                for j in range(x_tensor_2.size(0)):
                    label_zhr = np.append(label_zhr, 1.0 - abs(i - j)/64)
                    norm_2 = torch.norm(x_tensor_2[j])
                    similarity_score = torch.dot(x_tensor_1[i], x_tensor_2[j]) / (norm_1*norm_2)
                    similarity_numpy = np.append(similarity_numpy, similarity_score.item())
            # np.save('data/ntu/simility_numpy/array_{}_{}_after_cnn.npy'.format(index,index), similarity_numpy)
            # similarity_numpy_matrix = similarity_numpy.reshape(64, 64)


            # loss = self.loss(output_1, label_1)
            # backward
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

            # backward 用来尝试更新时间卷积层-start
            # print(torch.from_numpy(similarity_numpy_matrix).unsqueeze(0).unsqueeze(1))
            # out_put_zhr = self.Model_zhr_temp(torch.from_numpy(similarity_numpy_matrix).unsqueeze(0).unsqueeze(1).to(torch.float32))
            # label_zhr = torch.from_numpy(label_zhr).unsqueeze(0).unsqueeze(1).to(torch.float32)
            # out_put_zhr = out_put_zhr.flatten()
            # label_zhr = label_zhr.flatten()
            # print('label_zhr:', label_zhr)
            # loss_zhr = self.loss_temporary(out_put_zhr, label_zhr)
            # self.optimizer_zhr.zero_grad()
            # loss_zhr.backward()
            # self.optimizer_zhr.step()     
            # backward 用来尝试更新时间卷积层-end      

            # loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output_1.data, 1)
            acc = torch.mean((predict_label == label_1.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            # self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
            


        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        # result_tensor_list = {} #存放主干网，全连接层前的输出张量，最终存到文件里使用
        # result_tensor_list[0] = torch.randn(5,6,6,6)
        # torch.save(result_tensor_list, 'work_dir3/result_tensor_list/result_tensor_list.pth')
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                print('批量中的data: ',data.shape)
                print('批量中的index: ',index.shape)
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    x_tensor,output = self.model(data,label,index)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)


    def start(self):
        if self.arg.phase == 'train':
            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            # self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            # def count_parameters(model):
            #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
            # self.print_log(f'# Parameters: {count_parameters(self.model)}')
            # print('range大小',self.arg.start_epoch, self.arg.num_epoch)
            # for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            #     save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
            #             epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

            self.train(1, save_model=1)

            self.eval(1, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
