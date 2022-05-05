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
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from conf import log, log_dir, param
from dataset import *
from network import tsfusion


def train():
    """
    """

    dataset = param.dataset
    dataset_dir = param.dataset_dir
    dataset_preprocess_dir = param.dataset_preprocess_dir
    save_model_dir = param.save_model_dir
    model_name = param.model_name

    epoch_num = param.epoch_num
    resume_epoch = param.resume_epoch
    save_epoch_interval = param.snapshot
    test_interval = param.test_interval
    lr = param.lr

    model = tsfusion.TwoStreamFusion(class_num=101)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if multi_gpu_available():
        log.info("training on %d gpus" % torch.cuda.device_count())
        model = nn.DataParallel(model)

    log.info("training on %s device" % device)
    log.info("training model on %s dataset" % dataset)

    model_parameters = model.module.parameters() if multi_gpu_available() else model.parameters()
    optimizer = optim.SGD(model_parameters, lr=float(lr), momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch <= 0:
        log.info("training model (%s) from scratch" % model_name)
    else:
        load_checkpoint(save_model_dir, model_name, resume_epoch, model, optimizer)

    log.info("total params: %.2fM" % (sum(p.numel() for p in model_parameters) / 1000000.0))
    model.to(device)
    criterion.to(device)

    summary_log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=summary_log_dir)

    # define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataloader = DataLoader(Ucf101Dataset(dataset_dir, dataset_preprocess_dir, phase='train', transform=transform, preprocess=True),
                                  batch_size=6, shuffle=True, num_workers=0)

    test_dataloader = DataLoader(Ucf101Dataset(dataset_dir, dataset_preprocess_dir, phase='test', transform=transform, preprocess=False),
                                 batch_size=6, shuffle=True, num_workers=0)

    valid_dataloader = DataLoader(Ucf101Dataset(dataset_dir, dataset_preprocess_dir, phase='valid', transform=transform, preprocess=False),
                                  batch_size=6, shuffle=True, num_workers=0)

    iteration = 0
    train_valid_loaders = {'train': train_dataloader, 'valid': valid_dataloader}
    train_valid_sizes = {x: len(train_valid_loaders[x]) for x in ['train', 'valid']}
    test_size = len(test_dataloader)

    log.debug("train size: %d, test size: %d, valid size: %d" % (train_valid_sizes['train'], test_size, train_valid_sizes['valid']))

    for epoch in tqdm(range(resume_epoch, epoch_num), desc='epoch', leave=False):
        # each epoch has a training step and a validation step
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss: float = 0
            running_corrects: float = 0

            # set model to train() or eval() mode depending on whether it is trained or being validated
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for index, data in tqdm(enumerate(train_valid_loaders[phase]), desc=phase, leave=True):
                # move inputs and labels to the device the training is taking place on
                rgb_frames, optical_frames, labels = data

                rgb_frames = Variable(rgb_frames, requires_grad=True).to(device)
                optical_frames = Variable(optical_frames, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(rgb_frames, optical_frames)
                else:
                    with torch.no_grad():
                        outputs = model(rgb_frames, optical_frames)

                loss = criterion(outputs, labels.long())

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * rgb_frames.size(0)
                running_corrects += torch.sum(preds == labels.data)

                log.debug("[{}] epoch: {}/{}, lr: {}, loss: {}"
                          .format(phase, epoch + 1, epoch_num, optimizer.state_dict()['param_groups'][0]['lr'], running_loss))

            epoch_loss = running_loss / train_valid_sizes[phase]
            epoch_acc = running_corrects / train_valid_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            log.info("[{}] epoch: {}/{}, lr: {}, loss: {}, acc: {}, execution time: {}"
                     .format(phase, epoch + 1, epoch_num, optimizer.state_dict()['param_groups'][0]['lr'], epoch_loss, epoch_acc, stop_time - start_time))

        # save checkpoint
        if epoch % save_epoch_interval == 0 and iteration > 0:
            save_checkpoint(save_model_dir, model_name, epoch, model, optimizer)

        # test
        if epoch % test_interval == 0 and iteration > 0:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for index, data in tqdm(enumerate(test_dataloader), desc='test', leave=True):
                rgb_frames, optical_frames, labels = data

                rgb_frames = rgb_frames.to(device)
                optical_frames = optical_frames.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(rgb_frames, optical_frames)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * rgb_frames.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            stop_time = timeit.default_timer()
            log.info("[test] epoch: {}/{}, loss: {}, acc: {}, execution time: {}"
                     .format(epoch + 1, epoch_num, epoch_loss, epoch_acc, stop_time - start_time))

        iteration += 1

    writer.close()


def gpu_available() -> bool:
    return torch.cuda.is_available()


def gpu_count() -> int:
    return torch.cuda.device_count()


def multi_gpu_available() -> bool:
    return gpu_available() and gpu_count() > 1


def save_checkpoint(path_dir: str, model_name: str, epoch: int, model, optimizer):
    state = {
        'epoch': epoch,
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    path = model_path(path_dir, model_name, epoch)
    torch.save(state, path)
    log.info("save model at %s" % path)


def load_checkpoint(path_dir: str, model_name: str, epoch: int, model, optimizer):
    path = model_path(path_dir, model_name, epoch)
    log.info("loading model from: %s" % path)
    checkpoint = torch.load(path)

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])


def model_path(path_dir: str, model_name: str, epoch: int) -> str:
    return os.path.join(path_dir, "%s-epoch-%d.pth" % (model_name, epoch))


if __name__ == '__main__':
    train()
