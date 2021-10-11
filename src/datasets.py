import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import urllib.request
import shutil
import ssl
import bz2
from ftplib import FTP
import gzip
import scipy
import scipy.io

class BinaryDataset(Dataset):
    urls = {
        'titanic': 'ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/titanic.tar.gz',
        'covtype': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",
        'german': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer_scale",
        'diabetis': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale",
        'twonorm': 'ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/twonorm.tar.gz',
        'ringnorm': 'ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/ringnorm.tar.gz'
    }

    @staticmethod
    def _define_arch_type(arch_name):
        endings = ['.tar.gz', '.bz2']
        for ending in endings:
            if arch_name.endswith(ending):
                return arch_name[:-len(ending)], ending
        return arch_name, None

    @staticmethod
    def _unzip_arch(arch_path, dest_path, ending):
        if ending is None:
            if arch_path != dest_path:
                with open(arch_path, 'rb') as f_in:
                    with open(dest_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            return
        if ending == '.bz2':
            with bz2.open(arch_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return
        if ending == '.tar.gz':
            shutil.unpack_archive(arch_path, dest_path)
            return
        raise Exception(f"ending {ending} not defined")

    @staticmethod
    def _deine_url_type(url):
        url_types = ['ftp', 'https']
        for url_type in url_types:
            if url.startswith(url_type):
                return url_type
        raise Exception(f"url type of '{url}' not defined")

    @staticmethod
    def _load_arch_ftp(url, save_path):
        spl_url = url.split('/')
        assert spl_url[0] == 'ftp:'
        ftp_server_name = spl_url[2]
        file_path = os.path.join(save_path, spl_url[-1])
        if os.path.exists(file_path):
            return file_path
        ftp_handler = FTP(ftp_server_name)
        ftp_handler.login()
        server_file_path = '/'.join(spl_url[3:])
        with open(file_path, 'wb') as fp:
            ftp_handler.retrbinary(f'RETR {server_file_path}', fp.write)
        return file_path
    
    @staticmethod
    def _load_arch_https(url, save_path):
        assert url.split('/')[0] == 'https:'
        arch_file_name = url.split('/')[-1]
        file_path = os.path.join(save_path, arch_file_name)
        if os.path.exists(file_path):
            return file_path
        urllib.request.urlretrieve(url, file_path)
        return file_path

    def __init__(self, _type, data_path="data", unverified_ssl_enable=False):
        assert _type in ['titanic', 'covtype', 'german', 'diabetis', 'twonorm', 'ringnorm']
        self._type = _type
        self.unverified_ssl_enable = unverified_ssl_enable
        self.data_path = data_path
        if self.unverified_ssl_enable:
            ssl._create_default_https_context = ssl._create_unverified_context
        data_path = self._load_dataset()
        if self._type == 'titanic':
            self._prepare_titanic(data_path)
        elif self._type == 'covtype':
            self._prepare_covtype(data_path)
        elif self._type == 'german':
            self._prepare_standard_ds(data_path, 1000, 24)
        elif self._type == 'diabetis':
            self._prepare_standard_ds(data_path, 768, 8)
        elif self._type == 'twonorm':
            self._prepare_norm_ds(data_path, 'twonorm')
        elif self._type == 'ringnorm':
            self._prepare_norm_ds(data_path, 'ringnorm')
    
    def _prepare_standard_ds(self, file_path, n_items, n_features):
        self.n_features = n_features
        self.data = np.zeros((n_items, self.n_features))
        self.classes = np.zeros(n_items, dtype=np.int32)
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        for i_line, curr_line in enumerate(lines):
            line_split = curr_line.strip().split(' ')
            self.classes[i_line] = int(line_split[0])
            for _str in line_split[1:]:
                num, val = _str.split(':')
                num = int(num)
                val = float(val)
                self.data[i_line][num - 1] = val
    
    def _prepare_covtype(self, file_path):
        n_items = 581012
        self.n_features = 54
        self.data = np.zeros((n_items, self.n_features))
        self.classes = np.zeros(n_items, dtype=np.int32)
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        
        for i_line, curr_line in enumerate(lines):
            line_split = curr_line.strip().split(' ')
            self.classes[i_line] = 2 * int(line_split[0]) - 3 # (-1, 1) class labels
            for _str in line_split[1:]:
                num, val = _str.split(':')
                num = int(num)
                val = float(val)
                self.data[i_line][num - 1] = val
    
    def _prepare_titanic(self, file_path):
        ds_path = os.path.join(file_path, 'titanic', 'Source', 'titanic.dat')
        n_items = 2201
        self.n_features = 3
        self.data = np.zeros((n_items, self.n_features))
        self.classes = np.zeros(n_items, dtype=np.int32)
        with open(ds_path, 'r') as f:
            lines = f.read().splitlines()
        for i_line, curr_line in enumerate(lines):
            line_split = curr_line.strip().split()
            assert len(line_split) == 4
            self.classes[i_line] = 2 * int(line_split[-1]) - 1 # (-1, 1) class labels
            self.data[i_line, :] = np.asarray(list(map(float, line_split[:-1])))
    
    def _prepare_norm_ds(self, file_path, name):
        n_items = 7400
        self.n_features = 20
        ds_arch_path = os.path.join(file_path, name, 'Dataset.data.gz')
        self.data = np.zeros((n_items, self.n_features))
        self.classes = np.zeros(n_items, dtype=np.int32)
        with gzip.open(ds_arch_path, 'r') as f:
            lines = f.read().splitlines()
        for i_line, curr_line in enumerate(lines):
            line_split = curr_line.strip().split()
            assert len(line_split) == 21
            self.classes[i_line] = 2 * int(line_split[-1]) - 1 # (-1, 1) class labels
            self.data[i_line, :] = np.asarray(list(map(float, line_split[:-1])))

    def _load_dataset(self):
        if self.data_path != "":
            if not os.path.exists(self.data_path):
                os.mkdir(self.data_path)
        path = os.path.join(self.data_path, self._type)
        if not os.path.exists(path):
            os.mkdir(path)
        url = self.urls[self._type]
        url_type = self._deine_url_type(url)
        if url_type == 'ftp':
            arch_path = self._load_arch_ftp(url, path)
        elif url_type == 'https':
            arch_path = self._load_arch_https(url, path)
        file_path, ending = self._define_arch_type(arch_path)
        if not os.path.exists(file_path):
            self._unzip_arch(arch_path, file_path, ending)
        return file_path
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data_item = self.data[i]
        cls_item = self.classes[i]
        return data_item, cls_item

class GunnarRaetschBenchmarks:

    class _NpDataset(Dataset):

        def __init__(self, X, y):
            self.X = X
            self.y = y
        
        @property
        def n_features(self):
            return self.X.shape[1]
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, i):
            data_item = self.X[i]
            cls_item = self.y[i][0]
            return data_item, cls_item

    ds_names = [
        'banana', 
        'breast_cancer', 
        'diabetis', 
        'flare_solar', 
        'german',
        'heart', 
        'image', 
        'ringnorm', 
        'splice', 
        'thyroid', 
        'titanic', 
        'twonorm', 
        'waveform']

    def __init__(self, save_path='data'):
        url = "http://theoval.cmp.uea.ac.uk/matlab/benchmarks/benchmarks.mat"
        if save_path != "":
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        arch_file_name = url.split('/')[-1]
        file_path = os.path.join(save_path, arch_file_name)
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        self.datasets = scipy.io.loadmat(file_path)
    
    def get_dataset(self, name):
        assert name in self.ds_names
        X = self.datasets[name][0][0][0]
        y = self.datasets[name][0][0][1]
        return self._NpDataset(X, y)

def dataset2numpy(dataset):
    X = np.stack([dataset[i][0] for i in range(len(dataset))])
    y = np.stack([dataset[i][1] for i in range(len(dataset))])
    return X, y

def get_train_test_datasets(name, train_ratio=0.8, split=True, torch_split_rseed=42):
    _available_ds = [
        'covtype', 
        'german', 
        'diabetis', 
        'twonorm', 
        'ringnorm', 
        'banana', 
        'splice', 
        'waveform', 
        'image'] 
    assert name in _available_ds
    if name in ['covtype', 'diabetis', 'twonorm', 'ringnorm']:
        dataset = BinaryDataset(name, unverified_ssl_enable=True)
    if name in ['image', 'german', 'banana', 'splice', 'waveform']:
        dataset = GunnarRaetschBenchmarks().get_dataset(name)
    if not split:
        return dataset
    if torch_split_rseed is not None:
        torch.random.manual_seed(torch_split_rseed)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    # divide into train and test subsets
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, test_len])
    return dataset, train_ds, test_ds
