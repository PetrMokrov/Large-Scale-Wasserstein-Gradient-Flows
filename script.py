import argparse
from src.experiments import ou_icnn_jko_vary_dimensions_experiment, ou_em_vary_dimensions_experiment
from src.experiments import ou_icnn_jko_fixed_dimension_experiment
from src.experiments import convergence_mix_gauss_targ_experiment
from src.experiments import ou_dual_jko_vary_dimensions_experiment
from src.experiments import filtering_icnn_jko_experiment
from src.experiments import filtering_bbf_experiment, filtering_dual_jko_experiment
from src.experiments import conv_comp_icnn_jko_mix_gauss_targ_experiment
from src.experiments import conv_comp_em_mix_gauss_targ_experiment
from src.experiments import data_posterior_experiment
from src.experiments import ou_em_proxrec_fixed_dimension_experiment, ou_em_proxrec_vary_dimensions_experiment
from src.experiments import conv_comp_em_proxrec_mix_gauss_targ_experiment
import yaml

def extract_method(config, args):
    method = args.method
    methods_supported = list(config['method'].keys())
    assert method in methods_supported, f'method {method} not supported, consider supported list : {methods_supported}'
    dict_method = config.pop('method')[method]
    config['experiment_method'] = method
    return {**config, **dict_method}

def extract_exp_ns(config, args):
    exp_ns = args.exp_ns
    exps_count = config['exps_count']
    if exp_ns == -1:
        exp_ns = list(range(exps_count))
    else:
        assert isinstance(exp_ns, list)
        assert max(exp_ns) < exps_count, f'consider experiment numbers less, then {exps_count}'
        assert min(exp_ns) >= 0, f'experiment numbers must be non-negative'
    config['exp_numbers'] = exp_ns
    return config

def extract_verbose(config, args):
    config['verbose'] = args.verbose
    return config

def extract_device(config, args):
    config['device'] = args.device
    return config

parser = argparse.ArgumentParser(description='Runs our experiments')
parser.add_argument('experiment', help='experiment name')
parser.add_argument('--method', help='method solving the task (if needed)', type=str, default='default')
parser.add_argument('--exp_ns', type=int, nargs='+', help='number of experiments to peform (if needed)', default=-1)
parser.add_argument('--verbose', dest='verbose', action='store_const', const=True, default=False)
parser.add_argument('--device', action='store', help='device (for NN training)', type=str, default='cuda:0')

args = parser.parse_args()

experiment_map = {
    'diabetis_data_posterior':{
        'config_path': './configs/diabetis_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'german_data_posterior':{
        'config_path': './configs/german_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'splice_data_posterior':{
        'config_path': './configs/splice_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'banana_data_posterior':{
        'config_path': './configs/banana_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'waveform_data_posterior':{
        'config_path': './configs/waveform_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'ringnorm_data_posterior':{
        'config_path': './configs/ringnorm_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'twonorm_data_posterior':{
        'config_path': './configs/twonorm_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'image_data_posterior':{
        'config_path': './configs/image_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'covtype_data_posterior':{
        'config_path': './configs/covtype_data_posterior.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': data_posterior_experiment
    },
    'conv_comp_dim_2':{
        'config_path': './configs/convergence_comparison_dim_2.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    },
    'conv_comp_dim_4':{
        'config_path': './configs/convergence_comparison_dim_4.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    },
    'conv_comp_dim_6':{
        'config_path': './configs/convergence_comparison_dim_6.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    },
    'conv_comp_dim_8':{
        'config_path': './configs/convergence_comparison_dim_8.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    }, 
    'conv_comp_dim_10':{
        'config_path': './configs/convergence_comparison_dim_10.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    },
    'conv_comp_dim_12':{
        'config_path': './configs/convergence_comparison_dim_12.yml',
        'preprocess': [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': conv_comp_icnn_jko_mix_gauss_targ_experiment,
            'EM_sim_1000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_10000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_sim_50000': conv_comp_em_mix_gauss_targ_experiment,
            'EM_ProxRec_400': conv_comp_em_proxrec_mix_gauss_targ_experiment,
            'EM_ProxRec_1000': conv_comp_em_proxrec_mix_gauss_targ_experiment
        }
    },
    'ou_vary_dim_freq':{
        'config_path': './configs/ornstein_uhlenbeck_vary_dim.yml',
        'preprocess' : [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': ou_icnn_jko_vary_dimensions_experiment,
            'EM_sim_1000': ou_em_vary_dimensions_experiment, 
            'EM_sim_10000': ou_em_vary_dimensions_experiment, 
            'EM_sim_50000': ou_em_vary_dimensions_experiment,
            'dual_jko': ou_dual_jko_vary_dimensions_experiment,
            'EM_ProxRec_400': ou_em_proxrec_vary_dimensions_experiment,
            'EM_ProxRec_1000': ou_em_proxrec_vary_dimensions_experiment,
            'EM_ProxRec_10000': ou_em_proxrec_vary_dimensions_experiment
        }
    },
    'conv_mix_gauss_dim_13': {
        'config_path': './configs/convergence_mix_gauss_dim_13.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': convergence_mix_gauss_targ_experiment
    },
    'conv_mix_gauss_dim_64': {
        'config_path': './configs/convergence_mix_gauss_dim_64.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': convergence_mix_gauss_targ_experiment
    },
    'conv_mix_gauss_dim_32': {
        'config_path': './configs/convergence_mix_gauss_dim_32.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': convergence_mix_gauss_targ_experiment
    },
    'conv_mix_gauss_dim_128': {
        'config_path': './configs/convergence_mix_gauss_dim_128.yml',
        'preprocess': [extract_verbose, extract_device],
        'function': convergence_mix_gauss_targ_experiment
    },
    'filtering' : {
        'config_path': './configs/filtering.yml',
        'preprocess' : [extract_method, extract_exp_ns, extract_verbose, extract_device],
        'function': {
            'ICNN_jko': filtering_icnn_jko_experiment,
            'dual_jko': filtering_dual_jko_experiment,
            'bbf_1000': filtering_bbf_experiment,
            'bbf_10000': filtering_bbf_experiment,
            'bbf_50000': filtering_bbf_experiment,
            'bbf_100': filtering_bbf_experiment
        }
    }
}

exp_name = args.experiment
assert exp_name in experiment_map, f"Experiment '{exp_name}' not defined, consider one in the list: '{list(experiment_map.keys())}'"
tech_config = experiment_map[exp_name]
config_path = tech_config['config_path']
with open(config_path, 'r') as fp:
    config = yaml.full_load(fp)
for func in tech_config['preprocess']:
    config = func(config, args)
function = tech_config['function']
if callable(function):
    function(config)
elif isinstance(function, dict):
    poss_methods = list(function.keys())
    method = config['experiment_method']
    assert method in poss_methods, f"In experiment {exp_name}: method {method} not implemented, consider one in the list: '{poss_methods}'"
    function[method](config)
print('Done!')
    
