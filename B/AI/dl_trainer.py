# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import time
import psutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as ct
import settings
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False
from settings import logger, formatter
import struct
import models
import logging
import utils
from LR import LRSchedule
from encoding import huffman
#from tensorboardX import SummaryWriter
from datasets import DatasetHDF5
#writer = SummaryWriter()

import ptb_reader
import models.lstm as lstmpy
from torch.autograd import Variable
import json


torch.manual_seed(0)
if "run_threads" in os.environ:
    logger.info("Using %d CPU threads." % int(os.environ['run_threads']))
    torch.set_num_threads(int(os.environ['run_threads']))
else:
    logger.info("Default 4 CPU threads.")
    torch.set_num_threads(4)

_support_dataset = ['imagenet', 'cifar10']
_support_cnns = ['resnet20', 'resnet50', 'vgg16', 'vgg19', 'alexnet', 'inception_v3']

NUM_CPU_THREADS=1

process = psutil.Process(os.getpid())

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = 'mnistnet'

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def get_available_gpu_device_ids(ngpus):
    return range(0, ngpus)

def create_net(num_classes, dnn='resnet20', **kwargs):
    ext = None
    if dnn == 'resnet20':
        net = models.__dict__['resnet20'](num_classes=num_classes)
    elif dnn == 'resnet50':
        net = models.__dict__['resnet50'](num_classes=num_classes)
    elif dnn == 'mnistnet':
        net = MnistNet()
    elif dnn == 'vgg16':
        #net = models.VGG(dnn.upper())
        logger.info(models.__dict__.keys())
        net = models.__dict__['vgg16'](num_classes=num_classes)
        #net = torchvision.models.vgg16(num_classes=num_classes)
    elif dnn == 'alexnet':
        #net = models.AlexNet()
        net = torchvision.models.alexnet()
    elif dnn == 'inception_v3':
        net = torchvision.models.inception_v3(aux_logits=False)
    elif dnn == 'lstman4':
        net, ext = models.LSTMAN4(datapath=kwargs['datapath'])
    elif dnn == 'lstm':
        # model = lstm(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
        #              vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)
        net = lstmpy.lstm(vocab_size=kwargs['vocab_size'], batch_size=kwargs['batch_size'])

    else:
        errstr = 'Unsupport neural network %s' % dnn
        logger.error(errstr)
        raise errstr 
    return net, ext


class DLTrainer:

    # todo =================
    def __init__(self, rank, size, master='gpu10', dist=True, ngpus=1, batch_size=32, 
        is_weak_scaling=True, data_dir='./data', dataset='cifar10', dnn='resnet20', 
        lr=0.04, nworkers=1, prefix=None, sparsity=0.95, pretrain=None, num_steps=35):

        self.size = size
        self.rank = rank
        self.pretrain = pretrain
        self.dataset = dataset
        self.prefix=prefix
        self.num_steps = num_steps
        self.ngpus = ngpus
        if self.ngpus > 0:
            self.batch_size = batch_size * self.ngpus if is_weak_scaling else batch_size
        else:
            self.batch_size = batch_size
        self.is_cuda = self.ngpus > 0
        self.is_pinned = self.ngpus > 0
        if self.is_cuda:
            torch.cuda.manual_seed_all(3000)
        self.num_batches_per_epoch = -1
        if self.dataset == 'cifar10' or self.dataset == 'mnist':
            self.num_classes = 10
        elif self.dataset == 'imagenet':
            self.num_classes = 1000
        elif self.dataset == 'an4':
            self.num_classes = 29 
        # todo zhtang ==============
        elif self.dataset == 'ptb':
            self.num_classes = 10
        self.nworkers = nworkers # just for easy comparison
        # TODO zhtang =============
        self.data_dir = data_dir
        if type(dnn) != str:
            self.net = dnn
            self.dnn = dnn.name
            self.ext = None # leave for further parameters
        else:
            self.dnn = dnn
            if self.data_dir is not None and self.dnn != 'lstman4':
                self.data_prepare()
            # TODO: Refact these codes!
            if self.dnn == 'lstm':
                self.net, self.ext = create_net(self.num_classes, self.dnn, vocab_size = self.vocab_size, batch_size=self.batch_size)
            elif self.dnn == 'lstman4':
                self.net, self.ext = create_net(self.num_classes, self.dnn, datapath=self.data_dir)
                if self.data_dir is not None:  # an4, create net first and then prepare data
                    self.data_prepare()
            else:
                self.net, self.ext = create_net(self.num_classes, self.dnn)
        self.lr = lr
        self.base_lr = self.lr

        if self.is_cuda:
            if self.ngpus > 1:
                devices = get_available_gpu_device_ids(ngpus)
                self.net = torch.nn.DataParallel(self.net, device_ids=devices).cuda()
            else:
                self.net.cuda()
        #self.net.share_memory()
        self.accuracy = 0
        self.loss = 0.0
        self.train_iter = 0
        self.recved_counter = 0
        self.master = master
        self.average_iter = 0
        if self.dataset != 'an4':
            if self.is_cuda:
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            from warpctc_pytorch import CTCLoss
            self.criterion = CTCLoss()
            #from torch.nn import CTCLoss
            #self.criterion = CTCLoss(reduction='sum')
        self.lr_scheduler = getattr(LRSchedule, 'linear')(lr_init=self.lr, epochs=10, extra=0)
        weight_decay = 1e-4
        #if self.dataset == 'imagenet':
        #    weight_decay = 5e-4
        self.m = 0.9 # momentum
        nesterov = False
        if self.dataset != 'an4':
            nesterov = True
        else:
            self.lstman4_lr_epoch_tag = 0
            weight_decay = 0.
        self.optimizer = optim.SGD(self.net.parameters(), 
                lr=self.lr,
                #nesterov=True,
                momentum=self.m, 
                #weight_decay=5e-4)
                weight_decay=weight_decay,
                nesterov=nesterov)

        self.train_epoch = 0

        if self.pretrain is not None and os.path.isfile(self.pretrain):
            self.load_model_from_file(self.pretrain)

        self.avg_loss_per_epoch = 0.0
        self.timer = 0.0
        self.iotime = 0.0
        self.epochs_info = []
        self.distributions = {}
        self.gpu_caches = {}
        self.delays = []
        self.num_of_updates_during_comm = 0 
        self.train_acc_top1 = []
        logger.info('num_batches_per_epoch: %d'% self.num_batches_per_epoch)

    def get_acc(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_model_state(self):
        return self.net.state_dict()

    def get_data_shape(self):
        return self._input_shape, self._output_shape

    def get_train_epoch(self):
        return self.train_epoch

    def get_train_iter(self):
        return self.train_iter

    def set_train_epoch(self, epoch):
        self.train_epoch = epoch

    def set_train_iter(self, iteration):
        self.train_iter = iteration

    def load_model_from_file(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['state'])
        self.train_epoch = checkpoint['epoch']
        self.train_iter = checkpoint['iter']
        logger.info('Load pretrain model: %s, start from epoch %d and iter: %d', filename, self.train_epoch, self.train_iter)

    def get_num_of_training_samples(self):
        return len(self.trainset)

    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
        image_size = 224
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 1000)
        # inception_v3 need the input size to be 299*299, here ignore the convergence performance
        if self.dnn == 'inception_v3':
            scale = transforms.Scale(299)
            trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                scale,
                transforms.ToTensor(),
                normalize,
                ]))
        else:
            trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))
        self.trainset = trainset

        train_sampler = None
        shuffle = False
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=self.is_pinned, sampler=train_sampler)
        #testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
                transforms.ToPILImage(),
        #        transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=self.is_pinned)

    def cifar10_prepare(self):
        #transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #train_transform = transform
        #test_transform = transform
        image_size = 32
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        self.trainset = trainset
        self.testset = testset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=1)
        self.classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def mnist_prepare(self):
        image_size = 28
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        trainset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        self.trainset = trainset
        testset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        self.testset = testset
        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=self.batch_size, shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.batch_size, shuffle=False, num_workers=1)
    # todo zhtang =======
    def ptb_prepare(self):
        # Data loading code

        # =====================================
        # num_workers=NUM_CPU_THREADS num_workers=1
        # batch_size=self.batch_size
        # num_steps = 35
        # hidden_size = 1500

        # =================================
        raw_data = ptb_reader.ptb_raw_data(data_path=self.data_dir)
        train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
        self.vocab_size = len(word_to_id)


        self._input_shape = (self.batch_size, self.num_steps)
        self._output_shape = (self.batch_size, self.num_steps)

        print('Vocabluary size: {}'.format(self.vocab_size))

        print('load data')

        epoch_size = ((len(train_data) // self.batch_size) - 1) // self.num_steps

        train_set = ptb_reader.TrainDataset(train_data, self.batch_size, self.num_steps)
        self.trainset = train_set
        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=self.is_pinned, sampler=train_sampler)


        test_set = ptb_reader.TestDataset(valid_data, self.batch_size, self.num_steps)
        self.testset = test_set
        self.testloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=self.is_pinned)
        print('=========****** finish getting ptb data===========')

    def an4_prepare(self):
        from audio_data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
        audio_conf = self.ext['audio_conf']
        labels = self.ext['labels']
        train_manifest = os.path.join(self.data_dir, 'an4_train_manifest.csv')
        val_manifest = os.path.join(self.data_dir, 'an4_val_manifest.csv')


        with open('labels.json') as label_file:
            labels = str(''.join(json.load(label_file)))
        trainset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, labels=labels, normalize=True, augment=True)
        self.trainset = trainset
        testset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, labels=labels, normalize=True, augment=False)
        self.testset = testset

        if self.nworkers > 1:
            train_sampler = DistributedBucketingSampler(self.trainset, batch_size=self.batch_size, num_replicas=self.nworkers, rank=self.rank)
        else:
            train_sampler = BucketingSampler(self.trainset, batch_size=self.batch_size)

        self.train_sampler = train_sampler
        trainloader = AudioDataLoader(self.trainset, num_workers=8, batch_sampler=self.train_sampler)
        testloader = AudioDataLoader(self.testset, batch_size=self.batch_size,
                                  num_workers=8)
        self.trainloader = trainloader
        self.testloader = testloader


    def data_prepare(self):
        if self.dataset == 'imagenet':
            self.imagenet_prepare()
        elif self.dataset == 'cifar10':
            self.cifar10_prepare()
        elif self.dataset == 'mnist':
            self.mnist_prepare()
        elif self.dataset == 'an4':
            self.an4_prepare()
        elif self.dataset == 'ptb':
            self.ptb_prepare()
        else:
            errstr = 'Unsupport dataset: %s' % self.dataset
            logger.error(errstr)
            raise errstr
        self.data_iterator = None #iter(self.trainloader)
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)
        #self.num_batches_per_epoch = self.get_num_of_training_samples()/(self.batch_size*self.nworkers)

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer
        #self.trainloader = dataloader
        #self.data_iterator = iter(self.trainloader)
        #self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)/(self.batch_size*self.nworkers)
        ##self.num_batches_per_epoch = self.get_num_of_training_samples()/(self.batch_size*self.nworkers)
        #self.avg_loss_per_epoch = 0.0
        #self.timer = 0.0
        #self.epochs_info = []
        #logger.info('updated dataloader for SSGD, num_batches_per_epoch: %d'% self.num_batches_per_epoch)

    def update_nworker(self, nworkers, new_rank=-1):
        if new_rank >= 0:
            rank = new_rank
            self.nworkers = nworkers
        else:
            reduced_worker = self.nworkers - nworkers
            rank = self.rank
            if reduced_worker > 0 and self.rank >= reduced_worker:
                rank = self.rank - reduced_worker
        self.rank = rank
        # todo zhtang an4 ====================
        if self.dnn != 'lstman4':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainset, num_replicas=nworkers, rank=rank)
            train_sampler.set_epoch(self.train_epoch)
            shuffle = False
            self.train_sampler = train_sampler
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                      shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=1)
        self.nworkers = nworkers
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)

    def data_iter(self):
        try:
            d = self.data_iterator.next()
        except:
            self.data_iterator = iter(self.trainloader)
            d = self.data_iterator.next()
        
        #print(d[0].size())
        #print(d[0].size()[-1], self.batch_size)
        if d[0].size()[0] != self.batch_size:
            return self.data_iter()
        return d

    def _adjust_learning_rate_lstman4(self, progress, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def _adjust_learning_rate_general(self, progress, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr 

    def _adjust_learning_rate_vgg16(self, progress, optimizer):
        if progress > 0 and progress % 25 == 0:
            self.lr = self.base_lr / (2**(progress/25))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def adjust_learning_rate(self, progress, optimizer):
        #if self.dnn == 'vgg16':
        #    return self._adjust_learning_rate_vgg16(progress, optimizer)
        if self.dnn == 'lstman4':
           return self._adjust_learning_rate_lstman4(self.train_iter//self.num_batches_per_epoch, optimizer)        
        return self._adjust_learning_rate_general(progress, optimizer)

    def print_weight_gradient_ratio(self):
        #own_state = self.net.state_dict()
        #for name, param in own_state.items():
        # Tensorboard
        if self.rank == 0:
            for name, param in self.net.named_parameters():
                #writer.add_histogram(name, param.clone().cpu().data.numpy(), self.train_iter)
                writer.add_histogram(name, param.grad.clone().cpu().data.numpy(), self.train_iter)
        return
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                wn = param.data.norm()
                gn = param.grad.norm() 
                logger.info('[%s] = %f, %f, %f', name, wn, gn, wn/gn)

    def finish(self):
        writer.close()

    def cal_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        s = time.time()
        # todo ============
        # zero the parameter gradients
        #self.optimizer.zero_grad()
        for i in range(num_of_iters):
                # get the input
            # todo zhtang an4 =================================================
            self.adjust_learning_rate(self.train_epoch, self.optimizer)
            if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
                logger.info('train iter: %d, num_batches_per_epoch: %d', self.train_iter, self.num_batches_per_epoch)
                #self.adjust_learning_rate(self.train_epoch, self.optimizer)
                #logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
                aver_acc_top1 = -1
                if len(self.train_acc_top1) != 0:
                    aver_acc_top1 = torch.mean(torch.stack(self.train_acc_top1))
                logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, aver_acc_top1, self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
                    #sys.exit('NaN detected!!')
                self.train_acc_top1 = []
                #self.test(self.train_epoch)
                self.epochs_info.append(self.avg_loss_per_epoch/self.num_batches_per_epoch)
                self.avg_loss_per_epoch = 0.0
                self.train_epoch += 1
                # todo zhtang an4 ===========
                if self.train_sampler and (self.nworkers > 1):
                    # print(" In training :  self.train_sampler.set_epoch(self.train_epoch)  ")
                    self.train_sampler.set_epoch(self.train_epoch)

            ss = time.time()
            if data is None:
                data = self.data_iter()

            if self.dataset == 'an4':
                inputs, labels_cpu, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn == 'lstm' :
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
                else:
                    inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                if self.dnn == 'lstm' :
                    inputs = Variable(inputs.transpose(0, 1).contiguous())
                    labels = Variable(labels_cpu.transpose(0, 1).contiguous())
                else:
                    labels = labels_cpu
                
            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            #logger.info('[%d] labels: %s', self.train_iter, labels_cpu)
            self.iotime += (time.time() - ss)
            
            if self.dnn == 'lstman4':
                out, output_sizes = self.net(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                loss = self.criterion(out, labels_cpu, output_sizes, target_sizes)
                loss = loss / inputs.size(0)  # average the loss by minibatch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 400)
            elif self.dnn == 'lstm' :
                hidden = lstmpy.repackage_hidden(hidden)
                #print(inputs.size(), hidden[0].size(), hidden[1].size())
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size), tt)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.25)
                for p in self.net.parameters():
                    p.data.add_(-self.lr, p.grad.data)
            else:
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
            loss_value = loss.item()
            # logger.info statistics
            self.loss += loss_value 

            self.avg_loss_per_epoch += loss_value

            # todo zhtang an4 ==================
            if self.dnn not in ['lstm', 'lstman4']:
                acc1, = self.cal_accuracy(outputs, labels, topk=(1,))
                self.train_acc_top1.append(acc1)
                
            self.train_iter += 1
        self.num_of_updates_during_comm += 1
        self.loss /= num_of_iters 
        self.timer += time.time() - s 
        display = 10
        if self.train_iter % display == 0:
            logger.info('[%3d][%5d/%5d][rank:%d] loss: %.3f, average forward and backward time: %f, iotime: %f ' %
                  (self.train_epoch, self.train_iter, self.num_batches_per_epoch, self.rank,  self.loss, self.timer/display, self.iotime/display))
            mbytes = 1024.*1024
            if self.is_cuda:
                logger.info('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes', ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes)
            self.timer = 0.0
            self.iotime = 0.0
            if len(self.delays) > 0:
                delay = int(np.mean(self.delays))
            else:
                delay = 0
            logger.info('Delay interval: %d, average delay: %d', self.num_of_updates_during_comm- self.average_iter, delay)
            self.delays = []
            if self.is_cuda:
                torch.cuda.empty_cache()
            #self.print_weight_gradient_ratio()
            
        # todo zhtang====
        if self.dnn == 'lstm':
            return num_of_iters, hidden
        return num_of_iters

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = float(correct)/total
        loss = float(test_loss)/total
        logger.info('Epoch %d, lr: %f, val loss: %f, val acc: %f' % (epoch, self.lr, loss, acc))
        self.net.train()

    def update_model(self):
        self.optimizer.step()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_with_single(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, num_steps=1, num_iter=None, secs=-1, cuda_enable=True):
    if cuda_enable:
        torch.cuda.set_device(0)
        trainer = DLTrainer(0, nworkers, dist=False, batch_size=batch_size, 
            is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, 
            dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', num_steps = num_steps)
    else:
        trainer = DLTrainer(0, nworkers, dist=False, batch_size=batch_size, 
            is_weak_scaling=True, ngpus=-1, data_dir=data_dir, dataset=dataset, 
            dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', num_steps = num_steps)
        
    iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)

    times = []
    display = 5
    total_iters = 100
    if dnn == 'lstm':
        hidden = trainer.net.init_hidden()
    for i in range(20):
        s = time.time()
        trainer.optimizer.zero_grad()
        for j in range(nsteps_update):
            # todo zhtang ========
            if dnn == 'lstm':
                _, hidden = trainer.train(1, hidden=hidden)
            else:
                trainer.train(1)
        trainer.update_model()
        times.append(time.time()-s)
        if i % display == 0 and i > 0: 
            time_per_iter = np.mean(times)
            logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)

    if secs != -1:
        estimated_time = np.mean(times)
        total_iters = int(secs * 1.0/ estimated_time) + 1
        logger.info('Adjust %d iterations to meet %d secs.', total_iters, secs)

    times = []

    for i in range(total_iters):
        s = time.time()
        trainer.optimizer.zero_grad()
        for j in range(nsteps_update):
            # todo zhtang ========
            if dnn == 'lstm':
                _, hidden = trainer.train(1, hidden=hidden)
            else:
                trainer.train(1)
        trainer.update_model()
        times.append(time.time()-s)
        if i % display == 0 and i > 0: 
            time_per_iter = np.mean(times)
            logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10', 'an4', 'ptb'], help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20', 'vgg19', 'vgg16', 'alexnet', 'inception_v3', 'lstman4', 'lstm'], help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=10, help='Default maximum epochs to train')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument("--power-profile", help="enable power sampling", action="store_true")
    parser.add_argument("--num-iter", type=int, default=-1)
    parser.add_argument("--secs", type=int, default=-1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--dev-type", type=str, default="nvidia")
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    device_id = 0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
    relative_path = './logs/GPU_ID_%d/%s-n%d-bs%d-lr%.4f-ns%d' % (device_id, args.dnn, 1, batch_size, args.lr, args.nsteps_update)
    utils.create_path(relative_path)
    logfile = os.path.join(relative_path, settings.hostname+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if args.power_profile:
        # start power profiling
        pw_logfile = logfile.replace(".log", "-power.log")
        logger.info("Cool down GPU for 30 secs before executing the task...")
        os.system("nohup ./scripts/nvml_samples 1>%s 2>&1 &" % pw_logfile)
        time.sleep(5)

    num_iter = None
    cuda_enable = True
    if args.num_iter != -1:
        num_iter = args.num_iter
    if args.cuda == 0:
        cuda_enable = False
    train_with_single(args.dnn, args.dataset, args.data_dir, 1, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.num_steps, num_iter, args.secs, cuda_enable=cuda_enable)

    if args.power_profile:
        # kill power profiling
        logger.info("Cool down GPU for 30 secs after executing the task...")
        os.system("killall nvml_samples")
        time.sleep(5)
        pass
