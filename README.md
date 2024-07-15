# Statistical Reliability ML

This package provides implementations of Monte Carlo methods to estimate the probability of failure of neural networks under noisy inputs.

It takes as input a PyTorch model and samples from a dataset, and it outputs the probabilities of failure given the noise configuration.

It implements several estimators. Currently:

* a basic Crude Monte Carlo,
* the first-order reliability measure (FORM),
* an importance sampling Monte Carlo estimation.

The last two estimators require searching for the Most Probable Failure Point (MPP).
The package provides several methods to find these points.
One of these methods uses [Foolbox](https://foolbox.jonasrauber.de) adversarial attacks, an optional dependecncy for the package.

## Basic usage

Considering a PyTorch model ```model``` and a supervised datatet with a PyTorch tensor ```X_test``` of featrues and a tensor ```y_test``` of  true labels,
we configure a reliability experiment with the following steps.
s
### 1. Import the package

```python
import statistical_reliability_ml as strml
```

### 2. Create an experiment

```python
experiment = strml.StatsRelExperiment(model, X_test, y_test, x_min=0, x_max=1, n_rep=10, noise_dist='gaussian', epsilon_range=[0.3])
```

The parameters provided here are:

* ```x_min```, ```x_max```: the bounds for the samples features
* ```n_rep```: the number of repetitions for each experiment. Setting this parameter greater than 1 will allow to estimate the variance of the estimations.
* ```noise_dist```: the type of noise distribution (either Gaussian, uniform or real uniform noise).
* ```epsilon_range```: a list of scale values for the noise. Each value will be evaluated in a separate experiment.

Other parameters can be provided and are decribed in the description of the ```StatsRelExperiment``` class.

### 3. Configure an estimator

The estimators classes are available in the package ```estimators```.

For instance the Crude Monte Carlo estimator is configured in the following manner:

```python
method = strml.estimators.CrudeMC(N_samples=[100,1000,1e4], track_advs=True)
```

This estimator takes as parameter the number of samples to generate (```N_samples```). Such a parameter can either be a single value or a list of values. If it is list then several experiments will be performed, each using a different value for the parameter. If more parameters defined as a range of values, then the cross-product of all the parameters configurations will be tested.

The Boolean parameter ```track_advs``` set to True requires that random samples that are adversarial examples will be ouputed in the results of the experiment.

#### Optionaly, configure a MPP search method

The other estimators (FORM and Importance Sampling) require as a mandatory parameter a method to search for the Most Probable Failure Point (MPP). The MPP search methods are classes in the package ```mpp_search```.

For instance the Newton search is configured in the following manner:

```python
search_method = strml.mpp_search.MPPSearchNewton(max_iter=[100, 1e4, 1e5], real_mpp=True)
```

Again the parameter ```max_iter``` can either be a single value or a list of values.

Then, the importance sampling estimator for instance is instanciated like this:

```python
method = strml.estimators.ImportanceSampling(search_method, N_samples=1000 track_advs=False, save_weights=False, save_mpp=False)
```

### 4. Run the experiments

Given an experiment object and an estimator, we run all experiments with the following command:

```python
results_df, dict_out = experiment.run_estimation(method, test_indices=[0,1,5], verbose=2)
```

In this command ```test_indices``` is the list of indices from ```X_test``` that will be analyzed.

The outputs of the command are:

* ```results_df```: a Pandas DataFrame that store the results and the parameters of the experiments, with one line for each experiment.
* ```dict_out```: a dictionnary with additional results that depend on the configuration of the experiment and the estimator.
