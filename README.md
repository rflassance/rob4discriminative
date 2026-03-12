# rob4discriminative
Robustness quantification for probabilistic discriminative classifiers

This repository reproduces the analyses performed in the paper "Robustness Quantification for Discriminative Models: a New Robustness Metric and its Application to Dynamic Classifier Selection" submitted to the Conference on Uncertainty in Artificial Intelligence (UAI) 2026.

For the experiments performed in section 3.1, the Generative Forests (gefs) package is required. It can be found on GitHub at [AlCorreia/GeFs](https://github.com/AlCorreia/GeFs). Using the package requires Python to be at most at version 3.7.

All datasets used in the experiments can be found at the "datasets" folder. The Jupyter notebooks contain the code of the applications, while the .py are files with auxiliaty functions. The Jupyter notebooks are as follows:

- (gef)robustness_metrics: Reproduces the experiment in section 3.1.1

- (gef)robustness_of_models: Reproduces the experiment in section 3.1.2

- dynamic_selection: Reproduces the experiments in section 3.2

For convenience, each notebook is split in a section that performs the calculations and another in which the results are shown. At the beginning of the results section, it is possible to load the respective pickle file with the pertinent results pre-recorded.