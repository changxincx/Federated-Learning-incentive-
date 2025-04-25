import os
import collections
import logging
import glob
import re

import torch, torchvision
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data.dataset import Dataset

import itertools as it
import copy


#-------------------------------------------------------------------------------------------------------
# DATASETS
#-------------------------------------------------------------------------------------------------------
DATA_PATH = os.path.join('TRAINING_DATA', 'PyTorch')




def get_mnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True)
  data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True)

  x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)

  return x_train, y_train, x_test, y_test


def get_fashionmnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=True, download=True)
  data_test = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=False, download=True)

  x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)

  return x_train, y_train, x_test, y_test



def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))


#-------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
#-------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------生成噪音的
config = {
    'noise_level': 0.4,  # 默认噪声等级
    'noisy_client_start': 1979,  # 开始添加噪声的客户端ID
    'noisy_client_end': 1999  # 结束添加噪声的客户端ID
}


def add_salt_and_pepper_noise(images, amount=0.3):
  """为图像添加椒盐噪声"""
  # 椒盐噪声的数量表示图像中受影响的像素比例
  # 创建一个和图像同样大小的随机数组
  noise = np.random.rand(*images.shape)
  images[noise < amount / 2] = 0  # Pepper
  images[noise > 1 - amount / 2] = 1  # Salt
  return images
def add_noise_to_images(images, noise_level):
    """为图像添加高斯噪声"""
    noisy_images = images + noise_level * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0, 1)  # 保证图像数据在有效范围内
    return noisy_images




def split_image_data3(data, labels, n_clients=240, uniform_clients=210, min_classes=5, max_classes=16, shuffle=True, verbose=True):

    n_data = data.shape[0]
    n_labels = np.max(labels) + 1
    uniform_clients = 204
    min_classes = 5
    max_classes = 16
    max_samples_per_bad_client=10

    indices = np.arange(n_data)
    if shuffle:
        np.random.shuffle(indices)
        data, labels = data[indices], labels[indices]

    clients = []

    # 分配数据给非均匀客户端
    remaining_data = data
    remaining_labels = labels

    for _ in range(n_clients - uniform_clients):
        num_classes = np.random.randint(min_classes, max_classes + 1)
        selected_classes = np.random.choice(range(n_labels), size=num_classes, replace=False)

        client_data = []
        client_labels = []

        for cls in selected_classes:
            cls_indices = np.where(remaining_labels == cls)[0]
            num_samples = np.random.randint(1, min(len(cls_indices) // (n_clients - uniform_clients),
                                                   max_samples_per_bad_client) + 1)  # 随机选择1到max_samples_per_bad_client之间的数量

            if num_samples > 0:
                selected_indices = np.random.choice(cls_indices, num_samples, replace=False)

                client_data.extend(remaining_data[selected_indices])
                client_labels.extend(remaining_labels[selected_indices])

                # 从剩余数据中移除这些数据
                remaining_data = np.delete(remaining_data, selected_indices, axis=0)
                remaining_labels = np.delete(remaining_labels, selected_indices, axis=0)

        if len(client_data) == 0:
            continue

        clients.append((np.array(client_data), np.array(client_labels)))

    # 将剩余数据均分给均匀客户端
    if len(remaining_data) > 0:
        data_per_uniform_client = len(remaining_data) // uniform_clients
        for _ in range(uniform_clients):
            if len(remaining_data) == 0:
                break
            client_data = remaining_data[:data_per_uniform_client]
            client_labels = remaining_labels[:data_per_uniform_client]

            clients.append((client_data, client_labels))

            remaining_data = remaining_data[data_per_uniform_client:]
            remaining_labels = remaining_labels[data_per_uniform_client:]
    for i in range(36, 72):
        if len(clients[i][0]) > 0:  # 确保客户端有数据
            original_data = clients[i][0]
            noisy_data = add_salt_and_pepper_noise(original_data)
            clients[i] = (noisy_data, clients[i][1])


    if verbose:
        print("Data split:")
        for i, (client_data, client_labels) in enumerate(clients):
            class_distribution = np.bincount(client_labels.astype(int), minlength=n_labels)
            print(f" - Client {i}: Class distribution: {class_distribution}, Total samples: {len(client_labels)}")

    return clients


# 打印每个客户端的数据分割结果
# --------------------------------------------------------------------------------------按id去选择好的和坏的

def split_image_data_random(data, labels, n_clients=3000, classes_per_client=10, shuffle=False, verbose=True):
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1

  # 配置客户端数量
  num_good_clients = 700

  num_noisy_clients = 150  # 添加噪声的客户端数量
  num_bad_clients_late = 150  # 后期坏客户端数量

  # 数据量配置
  data_per_good_client = 42000 // num_good_clients
  data_per_noisy_client = 9000 // num_noisy_clients  # 噪声客户端每个分到的数据
  data_per_bad_client_late = 6000 // num_bad_clients_late

  indices = np.arange(n_data)
  if shuffle:
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

  clients = []
  index = 0


  # 分配数据给好的客户端 (100-1979)
  for _ in range(num_good_clients):
    if index + data_per_good_client > n_data:
      break
    client_data = data[index:index + data_per_good_client]
    client_labels = labels[index:index + data_per_good_client]
    clients.append((client_data, client_labels))
    index += data_per_good_client

  # 添加噪声的客户端 (1960-1980)
  for _ in range(num_noisy_clients):
    if index + data_per_noisy_client > n_data:
      break
    client_data = data[index:index + data_per_noisy_client]
    client_labels = labels[index:index + data_per_noisy_client]
    client_data = add_salt_and_pepper_noise(client_data)  # 为客户端数据添加椒盐噪声
    clients.append((client_data, client_labels))
    index += data_per_noisy_client

    # 分配数据给后期坏的客户端 (1980-1999)，随机选择1-4个类别
  for _ in range(num_bad_clients_late):
    if index + data_per_bad_client_late > n_data:
      break
    client_data = data[index:index + data_per_bad_client_late]
    client_labels = labels[index:index + data_per_bad_client_late]
    num_classes = np.random.randint(2, 5)  # 随机选择1-4个类别
    selected_classes = np.random.choice(range(n_labels), size=num_classes, replace=False)
    mask = np.isin(client_labels, selected_classes)
    skewed_data = client_data[mask]
    skewed_labels = client_labels[mask]
    if len(skewed_data) < 1:
      skewed_data = client_data[:10]  # 保证至少有数据
      skewed_labels = client_labels[:10]
    clients.append((skewed_data, skewed_labels))
    index += data_per_bad_client_late

  if verbose:
    print("Data split:")
    for i, (client_data, client_labels) in enumerate(clients):
      print(f" - Client {i}: Class distribution: {np.bincount(client_labels, minlength=n_labels)}")

  return clients

def get_traffic_sign():
  '''Return Traffic Sign train/test data and labels as numpy arrays'''
  data_folder = Path(__file__).resolve().parent / 'TRAINING_DATA' / 'data'
  with open(data_folder.joinpath('train.p'), mode='rb') as f:
    train_set = pickle.load(f)
  with open(data_folder.joinpath('valid.p'), mode='rb') as f:
    valid_set = pickle.load(f)
  with open(data_folder.joinpath('test.p'), mode='rb') as f:
    test_set = pickle.load(f)

  x_train, y_train = np.array(train_set['features']), np.array(train_set['labels'])
  x_valid, y_valid = np.array(valid_set['features']), np.array(valid_set['labels'])
  x_test, y_test = np.array(test_set['features']), np.array(test_set['labels'])

  # Combine train and valid sets
  x_train = np.concatenate((x_train, x_valid), axis=0)
  y_train = np.concatenate((y_train, y_valid), axis=0)

  x_train = np.transpose(x_train, (0, 3, 1, 2))
  x_test = np.transpose(x_test, (0, 3, 1, 2))

  return x_train, y_train, x_test, y_test

#-------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
#-------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]


def get_default_data_transforms(name, train=True, verbose=True):
  transforms_train = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'traffic_sign': transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.339, 0.311, 0.321), (0.272, 0.261, 0.268))]),
  }
  transforms_eval = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'traffic_sign': transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.339, 0.311, 0.321), (0.272, 0.261, 0.268))]),
  }

  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train[name].transforms:
      print(' -', transformation)
    print()

  return (transforms_train[name], transforms_eval[name])


def get_data_loaders(hp, verbose=True):

  x_train, y_train, x_test, y_test = globals()['get_'+hp['dataset']]()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(hp['dataset'], verbose=False)


  # split = split_image_data3(x_train, y_train) # GTSRB

  split = split_image_data_random(x_train, y_train, n_clients=hp['n_clients'],  # 7crf10数据集
         classes_per_client=hp['classes_per_client'], verbose=verbose)   # 1000个优质客户端是6.1000\200是5

  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                                batch_size=hp['batch_size'], shuffle=True) for x, y in split]
  train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, transforms_eval), batch_size=100, shuffle=True)
  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=True)

  stats = {"split" : [x.shape[0] for x, y in split]}

  return client_loaders, train_loader, test_loader, stats

