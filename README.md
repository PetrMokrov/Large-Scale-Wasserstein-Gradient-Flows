# Large-Scale Wasserstein Gradient Flows

This repository contains **code** for reproducing the experiments as well as **results** of the experiments related to our article  *[Large-Scale Wasserstein Gradient Flows](https://arxiv.org/abs/2106.00736)* accepted for NeurIPS 2021. In our work we approximate gradient flows and, in particular, diffusion processes governed by Fokker-Planck equation using so-called JKO scheme modelled via Input Convex Neural Networks. We propose four experiments which demonstrate that our approach capable to work in different scenarious and machine learning applications.

## Prerequisites

It is highly recommended to use GPU to launch our experiments. The list of required python libraries can be found in `./requirements.txt`. One can install the libraries via the following command:

```bash
> pip install -r requirements.txt
```

## Experiments

All our experiments could be launched via `./script.py` script. The experiments use config files presented in `./configs/` directory which define hyperparameters of the experiments. See our submission for the details.

The results of the experiments are saved to the `./results` directory and could be visualized using `./W2JKO_results.ipynb` notebook. All the images representing our experiments are stored in `./images` directory

### Convergence to Stationary Solution

We test if our gradient flow approximating advection-diffusion process manage to converge to the stationary distribution.

#### Quantitative comparison

To reproduce the quantitative comparison of our method with particle based methods run the following:

```bash
> python .\script.py conv_comp_dim_[dimensionality] --method [method] --device [device]
```

Use *D = 2, 4, 6, 8, 10, 12* for `dimensionality` option, available methods are `ICNN_jko`, `EM_sim_1000`, `EM_sim_1000`, `EM_sim_50000`. Additionally one can consier `EM_ProxRec_400` and `EM_ProxRec_1000` methods. The `device` option make sense only for `ICNN_jko` method. 

In particular, the command below launches quantitative comparison experiment for the dimension *D=8* using our method on the `cuda:0` device:

```bash
> python .\script.py conv_comp_dim_8 --method ICNN_jko --device cuda:0
```

The results for all dimensions are presented in the image below:

<img src="./images/convergence_comparison.png" alt="drawing" width="400"/>

#### Qualitative comparison

The qualitative comparion results could be reprodused via the following command:

```bash
> python .\script.py conv_mix_gauss_dim_[dimensionality] --device [device]
```

The `dimensionality` can be either *D = 13* or *D = 32*. The comparison between fitted and true stationary distribution for *D = 32* below:

<img src="./images/convergence_mix_gauss_32.png" alt="drawing" width="400"/>

### Modelling Ornstein-Uhlenbeck processes

We model advection-diffusion processes with special quadratic-form potentials which have close-form solution for marginal process distribution at each observation time.

To launch Ornstein-Uhlenbeck experiment run the command below:

```bash
> python .\script.py ou_vary_dim_freq --method [method] --device [device]
```
The available options for `method` are `ICNN_jko`, `EM_sim_1000`, `EM_sim_10000`, `EM_sim_50000`, `'EM_ProxRec_10000`, `dual_jko`. 

The obtained divergence between true and fitted distributions for *t = 0.9* sec.:

<img src="./images/ou_sym_kl_0_9.png" alt="drawing" width="400"/>

### Unnormalized Posterior Sampling

Given the prior distribution of model parameters and conditional data distribution we model posterior parameters distribution by establishing it as stationary one of the gradient flow.

The experiments with different benchmark datasets could be run as follows:

```bash
> python .\script.py [dataset]_data_posterior --device [device]
```
The supported datasets are : `covtype`, `diabetis`, `german`, `splice`, `banana`, `waveform`, `ringnorm`, `twonorm`, `image`.

### Nonlinear filtering

We model predictive distribution at final time-moment of the latent highly-nonlinear diffusion process *X* given noisy observations obtained at specific time moments.

To reproduce our results run the command:
```bash
> python .\script.py filtering --method [method] --device [device]
```
The available methods are: `ICNN_jko`, `dual_jko`, `bbf_100`, `bbf_1000`, `bbf_10000`, `bbf_50000`

The obtained discrepancy between fitted methods and ground truth method (Chang&Cooper numerical integration) presented below:

<img src="./images/filtering_boxplot.png" alt="drawing" width="400"/>

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing our paper.

```
@misc{mokrov2021largescale,
      title={Large-Scale Wasserstein Gradient Flows}, 
      author={Petr Mokrov and Alexander Korotin and Lingxiao Li and Aude Genevay and Justin Solomon and Evgeny Burnaev},
      year={2021},
      eprint={2106.00736},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

