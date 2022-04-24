# coding=utf-8

"""
@author: shenke
@project: c3d
@file: train.py
@date: 2021/4/20
@description: 
"""

import os
import timeit

from tqdm import tqdm

import cv2
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision.models.resnet as resnet

from tensorboardX import SummaryWriter

from conf import param
from conf import log
from utils import util
from network import tsfusion


def train():
    """
    """

    dataset = param.dataset
    dataset_dir = param.dataset_dir
    save_model_dir = param.save_model_dir

    epoch_num = param.epoch_num
    resume_epoch = param.resume_epoch
    save_epoch_interval = param.snapshot
    test_interval = param.test_interval
    lr = param.lr

    model_name = 'two-stream-fusion'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log.info('Training on %s.' % device)

    model = tsfusion.TwoStreamFusion()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if resume_epoch <= 0:
        log.info("Training %s from scratch." % model_name)
    else:
        load_checkpoint(save_model_dir, resume_epoch, model, optimizer)

    log.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print('Training model on %s dataset.' % dataset)
    # train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=6, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=6, num_workers=0)
    # valid_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=6, num_workers=0)

    iteration = 0

    for epoch in tqdm(range(resume_epoch, epoch_num)):
        model.train()

        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        # save checkpoint
        if epoch % save_epoch_interval == 0 and iteration > 0:
            save_checkpoint(save_model_dir, epoch, model, optimizer)

        # test
        if epoch % test_interval == 0 and iteration > 0:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        iteration += 1


def save_checkpoint(path_dir, epoch, model, optimizer):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    path = __model_path(path_dir, epoch)
    torch.save(state, path)
    log.info("Save model at %s." % path)


def load_checkpoint(path_dir, epoch, model, optimizer):
    path = __model_path(path_dir, epoch)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    log.info("Load para from: %s." % path)


def __model_path(path_dir, epoch):
    return os.path.join(path_dir, 'model-epoch-%d.pth' % epoch)


if __name__ == '__main__':
    train()
