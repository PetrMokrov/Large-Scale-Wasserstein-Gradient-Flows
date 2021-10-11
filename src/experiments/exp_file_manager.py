import json
from pathlib import Path
import os, sys
from copy import deepcopy
from collections import OrderedDict
from datetime import datetime
import pytz
import shutil
import dill
import torch
import numpy as np
import uuid
import yaml

class ExpFileManager:
    
    def __init__(self, dir_name, config, temporary=False):
        self.temporary = temporary
        self.config = deepcopy(config)
        self.dir_removed=False
        if not self.temporary:
            self.dir_name = dir_name
        else:
            timezone_str = "Africa/Casablanca"
            timestamp = datetime.now(
                pytz.timezone(timezone_str)).strftime("%Y_%m_%d_%H%M%S")
            uuid_tstp = str(uuid.uuid4())[:8] + '_' + timestamp
            dir_name = os.path.join(dir_name, uuid_tstp)
            if Path(dir_name).is_dir():
                raise Exception(f'Temporary directory "{dir_name}" already exists!')
            self.dir_name = dir_name
    
    @classmethod
    def fromconfig(cls, config, temporary=False):
        return cls(config['experiment_dir'], config, temporary=temporary)
    
    @classmethod
    def fromconfigpath(cls, config_path, dir_name=None, temporary=False):
        with open(config_path, 'r') as fp:
            config = yaml.full_load(fp)
        if dir_name is not None:
            return cls(dir_name, config, temporary=temporary)
        return cls.fromconfig(config, temporary=temporary)
    
    @staticmethod
    def _create_file_name(params_list, params_ord_dict, exp='.json'):
        file_name = '-'.join(
            params_list + \
            [f"{key}={val}" for key, val in params_ord_dict.items()]) + \
            f'{exp}'
        return file_name
    
    def make_params(self, *args, **kwargs):
        raise NotImplementedError()
    
    def make_model_params(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_params(*args, **kwargs)
        params_list = ['model',] + params_list
        return params_list, params_ord_dict
    
    def make_np_params(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_params(*args, **kwargs)
        params_list = ['np',] + params_list
        return params_list, params_ord_dict
    
    def save(self, to_save, *args, **kwargs):
        params_list, params_ord_dict = self.make_params(*args, **kwargs)
        return self._save(to_save, params_list, params_ord_dict, save_backend='json', exp='.json')
    
    def save_model(self, to_save, *args, **kwargs):
        params_list, params_ord_dict = self.make_model_params(*args, **kwargs)
        return self._save(to_save, params_list, params_ord_dict, save_backend='torch', exp='.pth')
    
    def save_np(self, to_save, *args, **kwargs):
        params_list, params_ord_dict = self.make_np_params(*args, **kwargs)
        return self._save(to_save, params_list, params_ord_dict, save_backend='np', exp='.npy')
    
    def _save(self, to_save, params_list, params_ord_dict, save_backend='json', exp='.json'):
        if self.dir_removed:
            assert not Path(self.dir_name).is_dir()
            raise Exception(f'directory "{self.dir_name}" was already removed!')
        Path(self.dir_name).mkdir(parents=True, exist_ok=True)
        file_name = self._create_file_name(params_list, params_ord_dict, exp)
        file_path = os.path.join(self.dir_name, file_name)
        if save_backend=='json':
            with open(file_path, 'w') as fp:
                json.dump(to_save, fp)
        elif save_backend=='torch':
            torch.save(to_save, file_path, pickle_module=dill)
        elif save_backend=='np':
            assert isinstance(to_save, np.ndarray), f'"np" backend can save only np.ndarrays, got {type(to_save)} instead'
            np.save(file_path, to_save)
        else:
            raise Exception(f"got unrecognized save_backend={save_backend}!")
    
    def load(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_params(*args, **kwargs)
        return self._load(
            params_list, params_ord_dict, save_backend='json', exp='.json')
    
    def load_model(self, *args, map_location=None, **kwargs):
        params_list, params_ord_dict = self.make_model_params(*args, **kwargs)
        self._torch_map_location = map_location
        return self._load(
            params_list, params_ord_dict, save_backend='torch', exp='.pth')
    
    def load_np(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_np_params(*args, **kwargs)
        return self._load(
            params_list, params_ord_dict, save_backend='np', exp='.npy')
    
    def _load(self, params_list, params_ord_dict, save_backend='json', exp='.json'):
        file_name = self._create_file_name(params_list, params_ord_dict, exp)
        file_path = os.path.join(self.dir_name, file_name)
        if not Path(file_path).is_file():
            raise Exception(f'file "{file_path}" not exists!')
        if save_backend == 'json':
            with open(file_path, 'r') as fp:
                results = json.load(fp)
            return results
        if save_backend == 'torch':
            results = torch.load(file_path, map_location=self._torch_map_location)
            return results
        if save_backend == 'np':
            results = np.load(file_path)
            return results
        raise Exception(f"got unrecognized save_backend={save_backend}")  
    
    def rm(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_params(*args, **kwargs)
        return self._rm(params_list, params_ord_dict, exp='.json')
    
    def rm_model(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_model_params(*args, **kwargs)
        return self._rm(params_list, params_ord_dict, exp='.pth')
    
    def rm_np(self, *args, **kwargs):
        params_list, params_ord_dict = self.make_np_params(*args, **kwargs)
        return self._rm(params_list, params_ord_dict, exp='.npy')
    
    def _rm(self, params_list, params_ord_dict, exp='.json'):
        file_name = self._create_file_name(params_list, params_ord_dict, exp)
        file_path = os.path.join(self.dir_name, file_name)
        if Path(file_path).is_file():
            Path(file_path).unlink()
        assert not Path(file_path).is_file()

    def rm_dir(self):
        if not self.temporary:
            raise Exception("Only temporary directories can be removed!")
        shutil.rmtree(self.dir_name, ignore_errors=True)
        self.dir_removed = True

class MethodedExpFileManager(ExpFileManager):
    
    @classmethod
    def fromconfigpath(cls, config_path, method, dir_name=None, temporary=False):
        with open(config_path, 'r') as fp:
            config = yaml.full_load(fp)
        methods_supported = list(config['method'].keys())
        assert method in methods_supported, f'method {method} not supported, consider supported list : {methods_supported}'
        dict_method = config.pop('method')[method]
        config['experiment_method'] = method
        config = {**config, **dict_method}
        if dir_name is not None:
            return cls(dir_name, config, temporary=temporary)
        return cls.fromconfig(config, temporary=temporary)

class MultiexpLocator(ExpFileManager):
    
    def loc(self, statistic_name):
        assert statistic_name in [
            'kl_sym', 'kl_train', 'kl_target', 
            'energy_based', 'time_train', 'time_est']
        statistic = []
        n_exps = self.config['exps_count']
        if statistic_name == 'kl_sym':
            for n_exp in range(n_exps):
                statistic.append(self.load(n_exp)['kl_train'] + self.load(n_exp)['kl_target'])
            return statistic
        for n_exp in range(n_exps):
            statistic.append(self.load(n_exp)[statistic_name])
        return statistic

class OU_fixed_dim_EFM(MethodedExpFileManager):
    
    def make_params(self, exp_number):
        params_list = [self.config['experiment_name'], self.config['experiment_method']]
        params_ord_dict = OrderedDict([
            ('dim', self.config['dim']), 
            ('n_exp', exp_number)])
        return params_list, params_ord_dict

class ConvergenceComparison_EFM(OU_fixed_dim_EFM, MultiexpLocator):
    pass

class OU_vary_dim_EFM(MethodedExpFileManager):
    
    def make_params(self, exp_number):
        params_list = [self.config['experiment_name'], self.config['experiment_method']]
        params_ord_dict = OrderedDict([
            ('dim_min', self.config['dim_min']), 
            ('dim_max', self.config['dim_max']), 
            ('n_exp', exp_number)])
        return params_list, params_ord_dict
    
    def loc(self, statistic_name, dim, n_step):
        assert statistic_name in [
            'kl_sym', 'kl_train', 'kl_target', 
            'energy_based', 'time_train', 'time_est']
        statistic = []
        n_exps = self.config['exps_count']
        for n_exp in range(n_exps):
            data = self.load(n_exp)
            get_stat = lambda _name: data[str(dim)][_name][n_step][-1]
            if statistic_name == 'kl_sym':
                statistic.append(get_stat('kl_train') + get_stat('kl_target'))
            else:
                statistic.append(get_stat(statistic_name))
        return statistic

class Convergence_EFM(ExpFileManager):
    
    def make_params(self, *args, **kwargs):
        params_list = [self.config['experiment_name']]
        params_ord_dict = OrderedDict()
        return params_list, params_ord_dict

class DataPosterior_EFM(ExpFileManager):

    def make_params(self, *args, **kwargs):
        params_list = [self.config['experiment_name']]
        params_ord_dict = OrderedDict()
        return params_list, params_ord_dict
    
    def loc(self, statistic_name):
        assert statistic_name in [
            'accuracy', 'log_lik', 'time_train', 'time_est']
        return self.load()[statistic_name]

class Filtering_EFM(MethodedExpFileManager, MultiexpLocator):
    
    def loc(self, statistic_name):
        #TODO: fix kl_targ -> kl_target in statistic name
        assert statistic_name in [
            'kl_sym', 'kl_train', 'kl_target', 
            'energy_based', 'time_train', 'time_est']
        statistic = []
        n_exps = self.config['exps_count']
        if statistic_name == 'kl_sym':
            for n_exp in range(n_exps):
                statistic.append(self.load(n_exp)['kl_train'] + self.load(n_exp)['kl_targ'])
            return statistic
        for n_exp in range(n_exps):
            statistic.append(self.load(n_exp)[statistic_name])
        return statistic
    
    def make_np_params(self, exp_number, data_name):
        params_list = ['np', self.config['experiment_name'], data_name]
        params_ord_dict = OrderedDict([('n_exp', exp_number),])
        return params_list, params_ord_dict
    
    def make_params(self, exp_number):
        params_list = [self.config['experiment_name'], self.config['experiment_method']]
        params_ord_dict = OrderedDict([('n_exp', exp_number),])
        return params_list, params_ord_dict
