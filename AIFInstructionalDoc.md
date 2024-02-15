# AIF360 Instructional Document

## Introduction to AIF360
AIF360 is an open-source Python library developed by IBM that provides tools for bias mitigation, fairness assessment, and transparency in AI systems. This document will guide you through the process of using AIF360 for your fairness-aware machine learning projects.

## Table of Contents
- [Installation](#installation)
- [Preprocessing Algorithms in AIF360](#preprocessing-algorithms-in-aif360)
  1. [Disparate Impact Remover](#disparate-impact-remover) 
  2. [Learning Fair Representation](#learning-fair-representation)
  3. [Optimized Preprocessing](#optimized-preprocessing)
  4. [Reweighing](#reweighing)
- [Inprocessing Algorithms in AIF360](#inprocessing-algorithms-in-aif360)
  1. [Adversarial Debiasing](#adversarial-debiasing)
  2. [ART Classifier](#art-classifier)
  3. [Gerry Fair Classifier](#gerry-fair-classifier)
  4. [Meta Fair Classifier](#meta-fair-classifier)
  5. [Prejudice Remover](#prejudice-remover)
  6. [Exponentiated Gradient Reduction](#exponentiated-gradient-reduction)
  7. [Grid Search Reduction](#grid-search-reduction)
- [Postprocessing Algorithms in AIF360](#postprocessing-algorithms-in-aif360)
  1. [Calibrated Equalized Odds](#calibrated-equalized-odds)
  2. [Equalized Odds](#equalized-odds)
  3. [Reject Option Classification](#reject-option-classification)


---
---
## Installation

### R

``` r
install.packages("aif360")
```

### Python

Supported Python Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.8 – 3.11     |
| Ubuntu  | 3.8 – 3.11     |
| Windows | 3.8 – 3.11     |

### (Optional) Create a virtual environment

AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. A virtual environment manager is strongly recommended to ensure dependencies may be installed safely. If you have trouble installing AIF360, try this first.

#### Conda

Conda is recommended for all configurations though Virtualenv is generally interchangeable for our purposes. [Miniconda](https://conda.io/miniconda.html) is sufficient (see [the difference between Anaconda and Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda) if you are curious) if you do not already have conda installed.

 - Then, to create a new Python 3.11 environment, run:

    ```bash
    conda create --name aif360 python=3.11
    conda activate aif360
    ```

 - The shell should now look like `(aif360) $`. To deactivate the environment, run:

    ```bash
    (aif360)$ conda deactivate
    ```

    The prompt will return to `$ `.

### Install with `pip`

- To install the latest stable version from PyPI, run:

    ```bash
    pip install aif360
    ```

    > Note: Some algorithms require additional dependencies (although the metrics will
    all work out-of-the-box). To install with certain algorithm dependencies
    included, run, e.g.:

    ```bash
    pip install 'aif360[LFR,OptimPreproc]'
    ```

 - or, for complete functionality, run:

    ```bash
    pip install 'aif360[all]'
    ```

    The options for available extras are: `OptimPreproc, LFR, AdversarialDebiasing,
DisparateImpactRemover, LIME, ART, Reductions, FairAdapt, inFairness,
LawSchoolGPA, notebooks, tests, docs, all`

    If you encounter any errors, try the [Troubleshooting](#troubleshooting) steps.

### Manual installation

 - Clone the latest version of the AIF repository:

    ```bash
    git clone https://github.com/Trusted-AI/AIF360
    ```

    If you'd like to run the examples, download the datasets now and place them in their respective folders as described in [aif360/data/README.md](aif360/data/README.md).

 - Then, navigate to the root directory of the project and run:

    ```bash
    pip install --editable '.[all]'
    ```

#### Run the Examples

- To run the example notebooks, complete the manual installation steps above. Then, if you did not use the `[all]` option, install the additional requirementsas follows:

    ```bash
    pip install -e '.[notebooks]'
    ```

    Finally, if you did not already, download the datasets as described in
    [aif360/data/README.md](aif360/data/README.md).

### Troubleshooting

If you encounter any errors during the installation process, look for yourissue here and try the solutions.

#### TensorFlow

- See the [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) page for detailed instructions.
 - Note: we require `'tensorflow >= 1.13.1'`.
 
Once tensorflow is installed, try re-running:
```bash
pip install 'aif360[AdversarialDebiasing]'
```

> TensorFlow is only required for use with the `aif360.algorithms.inprocessing.AdversarialDebiasing` class.

#### CVXPY

- On MacOS, you may first have to install the Xcode Command Line Tools if you never have installed them  previously:
    ```sh
    xcode-select --install
    ```

- On Windows, you may need to download the [Microsoft C++ Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
See the [CVXPY Install](https://www.cvxpy.org/install/index.html#mac-os-x-windows-and-linux) page for up-to-date instructions.

 - Then, try reinstalling via:

    ```bash
    pip install 'aif360[OptimPreproc]'
    ```

> CVXPY is only required for use with the`aif360.algorithms.preprocessing.OptimPreproc` class.


---
---
## Preprocessing Algorithms in AIF360

1. Disparate impact Remover
2. Learning fair Representations
3. Optimized Preprocessing
4. Reweighing

### 1. Disparate Impact Remover
**Class**: `aif360.algorithms.preprocessing.DisparateImpactRemover`(_repair_level=1.0, sensitive_attribute=''_)

**Description**: Disparate impact remover is a preprocessing technique that edits feature values to increase group fairness while preserving rank-ordering within groups.

**Initialization**:
```python
__init__(repair_level=1.0, sensitive_attribute='')
```

**Parameters**:
- `repair_level`(_float_): Repair amount. 0.0 is no repair while 1.0 is full repair.
- `sensitive_attribute`(_str_): Single protected attribute with which to do repair.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`


**Methods Explanation:**:

1. **`fit(dataset)`**: Train a model on the input
    ```python
    fit(dataset)
    ```
    **Parameters**: `dataset` (BinaryLabelDataset) - Dataset containing true labels.
    
    **Returns**: `DisparateImpactRemover` -  Returns self.

2. **`fit_predict`**: Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (BinaryLabelDataset) -  Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

3. **`fit_transform`**: Run a repairer on the non-protected features and return the transformed dataset.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (BinaryLabelDataset): Dataset that needs repair.
    
    **Returns**:`dataset (BinaryLabelDataset)`: Transformed Dataset.

4. **`predict`**: Return a new dataset with labels predicted by running this Transformer on the input.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (BinaryLabelDataset) -  Dataset containing labels that need to be transformed.
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.

5. **`transform`**: Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (BinaryLabelDataset) -  Dataset that needs to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Dataset with transformed `instance_weights` attribute.

> Note:
    In order to transform test data in the same manner as training data, the distributions of attributes conditioned on the protected attribute must be the same.

### 2. Learning Fair Representation

**Class**: `aif360.algorithms.preprocessing.LFR`(_unprivileged_groups, privileged_groups, k=5, Ax=0.01, Ay=1.0, Az=50.0, print_interval=250, verbose=0, seed=None_)

**Description**: Learning fair representations is a pre-processing technique that finds a latent representation which encodes the data well but obfuscates information about protected attributes.

**Initialization**:
```python
__init__(unprivileged_groups, privileged_groups, k=5, Ax=0.01, Ay=1.0, Az=50.0, print_interval=250, verbose=0, seed=None)
```

**Parameters**:
- `unprivileged_groups`(_tuple_): Representation for unprivileged group.
- `privileged_groups`(_tuple_): Representation for privileged group.
- `k`(_int, optional_): Number of prototypes.
- `Ax`(_float, optional_): Input reconstruction quality term weight.
- `Az`(_float, optional_): Fairness constraint term weight.
- `Ay`(_float, optional_): Output prediction error.
- `print_interval`(_int, optional_): Print optimization objective value every print_interval iterations.
- `verbose`(_int, optional_): If zero, then no output.
- `seed`(_int, optional_): Seed to make `predict` repeatable.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`

**Methods Explanation**:

1. **`fit(dataset)`**: Compute the transformation parameters that lead to fair representations.
    ```python
    fit(dataset, maxiter=5000, maxfun=5000)
    ```
    **Parameters**:
    - `dataset`(BinaryLabelDataset)- Dataset containing true labels.
    - `maxiter` (_int_) – Maximum number of iterations.
    - `maxfun` (_int_) – Maxinum number of function evaluations.
    
    **Returns**:`LFR` - Returns self.

2. **`fit_predict(dataset)`**: Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset (BinaryLabelDataset)` - Dataset containing labels that need to be transformed.
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.

3. **`fit_transform(dataset)`**: Fit and transform methods sequentially.
    ```python
    fit_transform(dataset, maxiter=5000, maxfun=5000, threshold=0.5)
    ```
    **Parameters**:
    - `dataset` (_BinaryLabelDataset_) – Dataset containing labels that needs to be transformed.
    - `maxiter` (_int_) – Maximum number of iterations.
    - `maxfun` (_int_) – Maxinum number of function evaluations.
    - `threshold` (_float, optional_) – threshold parameter used for binary label prediction.
    
    **Returns**: `BinaryLabelDataset` Transformed dataset.


4. **`predict(dataset)`**: Return a new dataset with labels predicted by running this Transformer on the input.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset (BinaryLabelDataset)` - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

5. **`transform(dataset)`**: Transform the dataset using learned model parameters.
    ```python
    transform(dataset, threshold=0.5)
    ```
    **Parameters**:`dataset (BinaryLabelDataset)` - Dataset containing labels that need to be transformed.
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.

  
### 3. Optimized Preprocessing

**Class**: `aif360.algorithms.preprocessing.OptimPreproc`(_optimizer, optim_options, unprivileged_groups=None, privileged_groups=None, verbose=False, seed=None_)

**Description**: Optimized preprocessing is a preprocessing technique that learns a probabilistic transformation that edits the features and labels in the data with group fairness, individual distortion, and data fidelity constraints and objectives.

**Initialization**:

- `__init__`: Initialize the `OptimPreproc` object with optimizer class, optimization options, and representation for privileged and unprivileged groups.
    ```python
    __init__(optimizer, optim_options, unprivileged_groups=None, privileged_groups=None, verbose=False, seed=None)
    ```


**Parameters**:
- `optimizer`: Optimizer class.
- `optim_options`: Options for optimization to estimate the transformation.
- `unprivileged_groups`: Representation for unprivileged group.
- `privileged_groups`: Representation for privileged group.
- `verbose`: Verbosity flag for optimization.
- `seed`: Seed to make fit and predict repeatable.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`


**Methods Explanation**:

1. **`fit`**(_dataset_): Compute optimal pre-processing transformation based on distortion constraint.
    ```python
    fit(dataset, sep='=')
    ```
    **Parameters**:
    - `dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    - `sep` (*str, optional*) – Separator for converting one-hot labels to categorical.    
    
    **Returns**:`OptimPreproc`- Returns self.

2. **`fit_predict`**(_dataset_): Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

3. **`fit_transform`**(_dataset, sep='=', transform_Y=True_): Perform fit() and transform() sequentially.
    ```python
    fit_transform(dataset, sep='=', transform_Y=True)
    ```
    **Parameters**:
    - `dataset` (_BinaryLabelDataset_): Dataset containing labels that need to be transformed.
    - `sep` (_str, optional_): Separator for converting one-hot labels to categorical.
    - `transform_Y` (_bool_): Flag that mandates transformation of Y (labels).

    **Returns**: `BinaryLabelDataset`  - Transformed dataset.
    
    > Note: Perfom `fit()` and `transform()` sequentially.

4. **`predict`**: Return a new dataset with labels predicted by running this Transformer on the input.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) -  Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` Transformed dataset.

5. `transform`: Transform the dataset to a new dataset based on the estimated transformation.
    ```python
    transform(dataset, sep='=', transform_Y=True)
    ```
    **Parameters**:
    - `dataset` (_BinaryLabelDataset_): Dataset containing labels that need to be transformed.
    - `sep` (_str, optional_): Separator for converting one-hot labels to categorical.
    - `transform_Y` (_bool_): Flag that mandates transformation of Y (labels).

    **Returns**: `BinaryLabelDataset`  - Transformed dataset.
    
> Note: This algorithm does not use the privileged and unprivileged groups that are specified during initialization yet. Instead, it automatically attempts to reduce statistical parity difference between all possible combinations of groups in the dataset


### 4. Reweighing

**Class**: `aif360.algorithms.preprocessing.Reweighing`(_unprivileged_groups, privileged_groups_)

**Description**: A preprocessing technique that assigns different weights to examples in each (group, label) combination to ensure fairness before classification.

**Initialization**:

- `__init__`: Initialize the `Reweighing` object with representation for privileged and unprivileged groups.

    ```python
    __init__(unprivileged_groups, privileged_groups)
    ```

**Parameters**
- `unprivileged_groups` (list(dict)): Representation for unprivileged group.
- `privileged_groups` (list(dict)): Representation for privileged group.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`


**Methods Explanation**

1. **`fit`**: Compute the weights for reweighing the dataset.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`Reweighing`: Returns self.

2. **`fit_predict`**: Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset`: Transformed dataset.

3. **`fit_transform`**: Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```

    **Parameters**: `dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

4. **`predict`**: Return a new dataset with labels predicted by running this Transformer on the input.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset`- Transformed dataset.

5. **`transform`**: Transform the dataset to a new dataset based on the estimated transformation.
    ```python
    transform(dataset)
    ```

    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset that needs to be transformed.
    
    **Returns**:`BinaryLabelDataset`: Dataset with transformed `instance_weights` attribute.

----
----
## Inprocessing Algorithms in AIF360

1. Adversarial Debiasing	
2. ART Classifier	
3. Gerry Fair Classifier	
4. Meta Fair Classifier	
5. Prejudice Remover	
6. Exponentiated Gradient Reduction	
7. Grid Search Reduction


### 1. Adversarial Debiasing
**class**: `aif360.algorithms.inprocessing.AdversarialDebiasing`(_unprivileged_groups, privileged_groups, scope_name, sess, seed=None, adversary_loss_weight=0.1, num_epochs=50, batch_size=128, classifier_num_hidden_units=200, debias=True_)

**Description**: Adversarial debiasing is an in-processing technique that learns a classifier to maximize prediction accuracy and simultaneously reduce an adversary’s ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the predictions cannot carry any group discrimination information that the adversary can exploit.

**Initialization:**
- `__init__`: Initialize the `Adversial Debiasing` object.

    ```python
    __init__(unprivileged_groups, privileged_groups, scope_name, sess, seed=None, adversary_loss_weight=0.1, num_epochs=50, batch_size=128, classifier_num_hidden_units=200, debias=True)
    ```

**Parameters:**
- `unprivileged_groups` (_tuple_) – Representation for unprivileged groups.
- `privileged_groups` (_tuple_) – Representation for privileged groups.
- `scope_name` (_str_) – Scope name for the TensorFlow variables.
- `sess` (_tf.Session_) – TensorFlow session.
- `seed` (_int, optional_) – Seed to make predictions repeatable.
- `adversary_loss_weight` (_float, optional_) – Hyperparameter that chooses the strength of the adversarial loss.
- `num_epochs` (_int, optional_) – Number of training epochs.
- `batch_size` (_int, optional_) – Batch size.
- `classifier_num_hidden_units` (_int, optional_) – Number of hidden units in the classifier model.
- `debias` (_bool, optional_) – Learn a classifier with or without debiasing.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`

**Methods Explanation:**

1. **`fit`**: Compute the model parameters of the fair classifier using gradient descent.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`Reweighing`: Returns self.

2. **`fit_predict`**: Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset`: Transformed dataset.

3. **`fit_transform`**: Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```

    **Parameters**: `dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

4. **`predict`**: Obtain the predictions for the provided dataset using the fair classifier learned.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset`- Transformed dataset.

5. **`transform`**: Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```

    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset that needs to be transformed.
    
    **Returns**:`BinaryLabelDataset`: Dataset with transformed `instance_weights` attribute.
    
### 2. ART Classifier

**Class**: `aif360.algorithms.inprocessing.ARTClassifier`(_art_classifier_)

**Description**: Wraps an instance of an ART classifier (`art.classifiers.Classifier`) to extend the `Transformer` interface.

**Initialization**:
- `__init__`: Initialize ARTClassifier.

```python
__init__(art_classifier)
```

**Parameters**:
  - `art_classifier` (_art.classifier.Classifier_) – A Classifier object from the adversarial-robustness-toolbox.

**Methods**:
- `fit`
- `fit_predict`
- `fit_transform`
- `predict`
- `transform`

**Methods Explanation:**
1. **`fit`**: Train a classifer on the input.
    ```python
    fit(dataset, batch_size=128, nb_epochs=20)
    ```
    **Parameters**:
    - `dataset` (_dataset_) – Training dataset.
    - `batch_size` (_int_) – Size of batches (passed through to ART).
    - `nb_epochs` (_int_) – Number of epochs to use for training (passed through to ART).
    
    **Returns**: `ARTClassifier` – Returns self.

2. **`fit_predict`**: Train a model on the input and predict the labels.
    ```python
    predict(dataset)
    ```
    **Parameters**: `dataset` (_dataset_) – Training dataset.
    
    **Returns**:`Dataset`- Dataset with labels predicted by running this Transformer on the input.

3. **`fit_transform`**: Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**: `dataset` (_dataset_) – Training dataset.
    
    **Returns**:`Dataset`- Transformed dataset.

4. **`predict`**:	Perform prediction for the input.
    ```python
    predict(dataset, logits=False)
    ```
    **Parameters**:
    - `dataset` (_dataset_) – Test dataset.
    - `logits` (_bool, optional_) – True if prediction should be done at the logits layer (passed through to ART).
    
    **Returns**: `Dataset` – Dataset with predicted labels in the labels field.

5. **`transform`**: Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_): Dataset containing True Labels.
    
    **Returns**:`Dataset`: Transformed Dataset.



### 3. Gerry Fair Classifier

**Class**: `aif360.algorithms.inprocessing.GerryFairClassifier`(_C=10, printflag=False, heatmapflag=False, heatmap_iter=10, heatmap_path='.', max_iters=10, gamma=0.01, fairness_def='FP', predictor=LinearRegression()_)

**Description**: An algorithm for learning fair classifiers with respect to rich subgroups. It supports fairness notions such as false positive, false negative, and statistical parity rates.
Rich subgroups are defined by (linear) functions over the sensitive attributes, and fairness notions are statistical: false positive, false negative, and statistical parity rates. This implementation uses a max of two regressions as a cost-sensitive classification oracle, and supports ***linear regression, support vector machines, decision trees, and kernel regression***

**Initialization**:
```python
__init__(C=10, printflag=False, heatmapflag=False, heatmap_iter=10, heatmap_path='.', max_iters=10, gamma=0.01, fairness_def='FP', predictor=LinearRegression())
```
**Parameters**
- `C` – Maximum L1 Norm for the Dual Variables (hyperparameter)
- `printflag` – Print Output Flag
- `heatmapflag` – Save Heatmaps every heatmap_iter Flag
- `heatmap_iter` – Save Heatmaps every heatmap_iter
- `heatmap_path` – Save Heatmaps path
- `max_iters` – Time Horizon for the fictitious play dynamic.
- `gamma` – Fairness Approximation Paramater
- `fairness_def` – Fairness notion, FP, FN, SP.
- `errors` – see fit()
- `fairness_violations` – see fit()
- `predictor` – Hypothesis class for the Learner. Supports LR, SVM, KR, Trees.

**Methods**:
- `fit`
- `fit_predict`	
- `fit_transform`	
- `generate_heatmap`
- `pareto`
- `predict`	
- `print_outputs`
- `save_heatmap`
- `transform`	

**Methods Explanation**:
1. **`fit`**:	Run Fictitious play to compute the approximately fair classifier.
    ```python
    fit(dataset, early_termination=True)
    ```
    **Parameters**:
    - `dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    - `early_termination` (_bool_) – Terminate Early if Auditor can’t find fairness violation of more than gamma.
    
    **Returns**: self
    
2. **`fit_predict`**:	Train a model on the input and predict the labels.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    
    **Returns**:`dataset_new` (_Dataset_) – Modified dataset object with predicted labels.
    
3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    
    **Returns**:`dataset_new` (_Dataset_) – Transformed dataset.
    
4. **`generate_heatmap`**:	Helper Function to generate the heatmap at the current time.
    ```python
    generate_heatmap(dataset, predictions, vmin=None, vmax=None, cols_index=[0, 1], eta=0.1)
    ```
    **Parameters**:
    - `dataset` (*Dataset*) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    - `iteration` - Current Iteration
    - `predictions` (*numpy.ndarray*) – Predictions of the model self on dataset.
    - `vmin` and `vmax` are parameters that are often used in visualization functions to define the range of values for color mapping. They control the color scale of the heatmap, where each color corresponds to a certain value in the data.
    
    **Returns**: None
5. **`pareto`**:	Assumes Model has FP specified for the metric. Trains for each value of gamma, returns error, FP (via training), and FN (via auditing) values.
    ```python
    pareto(dataset, gamma_list)
    ```
    **Parameters**:
    - `dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    - `gamma_list` (_list_) – List of gamma values to generate the pareto curve.

    **Returns**:
    - `errors` (_list_) – List of errors for each model.
    - `fp_violations` (_list_) – List of false positive violations of those models.
    - `fn_violations` (_list_) – List of false negative violations of those models.
    
    
6. **`predict`**:	Return dataset object where labels are the predictions returned by the fitted model.
    ```python
    predict(dataset, threshold=0.5)
    ```
    **Parameters**:
    - `dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    - `threshold` (_float_) – The positive prediction cutoff for the soft-classifier.

    **Returns**:
    - `dataset_new` (_Dataset_) – Modified dataset object where the labels attribute are the predictions returned by the self model.
    
    
7. **`print_outputs`**:	Helper function to print outputs at each iteration of fit.
    ```python
    print_outputs(iteration, error, group)
    ```
    **Parameters**:
    - `iteration` (_int_) – Current iteration.
    - `error` – Most recent error.
    - `group` – Most recent group found by the auditor.

    **Returns**: None
8. **`save_heatmap`**:	Helper Function to save the heatmap.
    ```python
    save_heatmap(iteration, dataset, predictions, vmin, vmax)
    ```
    **Parameters**:
    - `iteration` (_int_) – Current iteration.
    - `dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    - `predictions` (_numpy.ndarray_) – Predictions of the model self on dataset.
    - `vmin` and `vmax` are parameters that are often used in visualization functions to define the range of values for color mapping. They control the color scale of the heatmap, where each color corresponds to a certain value in the data.

    **Returns**: (`vmin, vmax`) (_tuple_)
9. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**: `dataset` (_Dataset_) – Dataset object with its own class definition in datasets folder inherits from class StandardDataset.
    
    **Returns**:`dataset_new` (_Dataset_) – Transformed dataset.


### 4. Meta Fair Classifier

**Class**: `aif360.algorithms.inprocessing.MetaFairClassifier`(_tau=0.8, sensitive_attr='', type='fdr', seed=None_)

**Description**: The meta algorithm here takes the fairness metric as part of the input and returns a classifier optimized w.r.t. that fairness metric.

**Initialization**:
```python
__init__(tau=0.8, sensitive_attr='', type='fdr', seed=None)
```

**Parameters**:
- `tau` (_double, optional_) – Fairness penalty parameter.
- `sensitive_attr` (_str, optional_) – Name of protected attribute.
- `type` (_str, optional_) – The type of fairness metric to be used. Currently “fdr” (false discovery rate ratio) and “sr” (statistical rate/disparate impact) are supported. To use another type, the corresponding optimization class has to be implemented.
- `seed` (_int, optional_) – Random seed.
    
**Methods**:
- `fit`	
- `fit_predict`	
- `fit_transform`	
- `predict`	
- `transform`	

**Methods Explanation**:
1. **`fit`**:	Learns the fair classifier.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) -  Dataset containing true labels.
    
    **Returns**:`MetaFairClassifier` - Returns self.
    
2. **`fit_predict`**:	Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
4. **`predict`**:	Obtain the predictions for the provided dataset using the learned classifier model.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset that needs to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

### 5. Prejudice Remover

**Class**: `aif360.algorithms.inprocessing.PrejudiceRemover`(_eta=1.0, sensitive_attr='', class_attr=''_)

**Description**: Prejudice remover is an in-processing technique that adds a discrimination-aware regularization term to the learning objective 

**Initialization**:
```python
__init__(eta=1.0, sensitive_attr='', class_attr='')
```
**Parameters**:
 - `eta` (_double, optional_): fairness penalty parameter
 - `sensitive_attr` (_str, optional_): name of the protected attribute
 - `class_attr` (_str, optional_): label name
    
**Methods**:
- `fit`	
- `fit_predict`	
- `fit_transform`	
- `predict`	
- `transform`	

**Methods Explanation**:
1. **`fit`**:	Learns the regularized logistic regression model.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_): Dataset containing true labels.
    
    **Returns**:`PrejudiceRemover` - Returns self.
    
2. **`fit_predict`**: Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
4. **`predict`**: Obtain the predictions for the provided dataset using the learned prejudice remover model.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset that needs to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.


## 6. Exponentiated Gradient Reduction
**class**: `aif360.algorithms.inprocessing.ExponentiatedGradientReduction`(_estimator, constraints, eps=0.01, max_iter=50, nu=None, eta0=2.0, run_linprog_step=True, drop_prot_attr=True_)

**Description**:
 - Exponentiated gradient reduction for fair classification. 
 - Exponentiated gradient reduction is an in-processing technique that reduces fair classification to a sequence of cost-sensitive classification problems, returning a randomized classifier with the lowest empirical error subject to fair classification constraints.

**Initialization**:
```python
__init__(estimator, constraints, eps=0.01, max_iter=50, nu=None, eta0=2.0, run_linprog_step=True, drop_prot_attr=True)
```
**Parameters**:	
 - `estimator` – An estimator implementing methods `fit(X, y, sample_weight)` and `predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and `sample_weight` is a vector of weights; labels y and predictions returned by `predict(X)` are either 0 or 1 – e.g. scikit-learn classifiers.
 - `constraints` (_str_ or _fairlearn.reductions.Moment_) – If string, keyword denoting the `fairlearn.reductions.Moment` object defining the disparity constraints – e.g., “_DemographicParity_” or “_EqualizedOdds_”. For a full list of possible options see `self.model.moments`. Otherwise, provide the desired `Moment` object defining the disparity constraints.
 - `eps` – Allowed fairness constraint violation; the solution is guaranteed to have the error within `2*best_gap` of the best error under constraint eps; the constraint violation is at most `2*(eps+best_gap)`.
 - `T` – Maximum number of iterations.
 - `nu` – Convergence threshold for the duality gap, corresponding to a conservative automatic setting based on the statistical uncertainty in measuring classification error.
 - `eta_mul` – Initial setting of the learning rate.
 - `run_linprog_step` – If True each step of exponentiated gradient is followed by the saddle point optimization over the convex hull of classifiers returned so far.
 - `drop_prot_attr` – Boolean flag indicating whether to drop protected attributes from training data.


**Methods**:
 - `fit`
 - `fit_predict`	
 - `fit_transform`	
 - `predict`	
 - `transform`	

**Methods Explanation**:
 1. **`fit`**:	Learns randomized model with less bias
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (BinaryLabelDataset) - Dataset containing true labels.
    
    **Returns**:`ExponentiatedGradientReduction` - Returns self.
 2. **`fit_predict`**:	Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
 3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing true labels.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
 4. **`predict`**:	Obtain the predictions for the provided dataset using the randomized model learned.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

 5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset containing labels that need to be transformed.
    
    **Returns**:`BinaryLabelDataset`- Transformed dataset.

## 7. Grid Search Reduction
**class**: `aif360.algorithms.inprocessing.GridSearchReduction`(_estimator, constraints, prot_attr=None, constraint_weight=0.5, grid_size=10, grid_limit=2.0, grid=None, drop_prot_attr=True, loss='ZeroOne', min_val=None, max_val=None_)

**Description**:
 - Grid search reduction for fair classification or regression.
 - Grid search is an in-processing technique that can be used for fair classification or fair regression. For classification, it reduces fair classification to a sequence of cost-sensitive classification problems, returning the deterministic classifier with the lowest empirical error subject to fair classification constraints  among the candidates searched. For regression, it uses the same principle to return a deterministic regressor with the lowest empirical error subject to the constraint of bounded group loss.

**Initialization**:
```python
__init__(estimator, constraints, prot_attr=None, constraint_weight=0.5, grid_size=10, grid_limit=2.0, grid=None, drop_prot_attr=True, loss='ZeroOne', min_val=None, max_val=None)
```

**Parameters**:	
 - `estimator` (_estimator_): An estimator implementing methods `fit(X, y, sample_weight)` and `predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and `sample_weight` is a vector of weights; labels `y` and predictions returned by `predict(X)` are either 0 or 1 – e.g., scikit-learn classifiers/regressors.
 - `constraints` (_str_ or _fairlearn.reductions.Moment_): If string, keyword denoting the `fairlearn.reductions.Moment` object defining the disparity constraints – e.g., “_DemographicParity_” or “_EqualizedOdds_”. For a full list of possible options see `self.model.moments`. Otherwise, provide the desired Moment object defining the disparity constraints.
 - `prot_attr` (_str_ or _array-like_): Column indices or column names of protected attributes.
 - `constraint_weight` (_float_): When the `selection_rule` is “_tradeoff_optimization_” (default, no other option currently) this float specifies the relative weight put on the constraint violation when selecting the best model. 
 The weight placed on the error rate will be `1-constraint_weight`.
 - `grid_size` (_int_): The number of Lagrange multipliers to generate in the grid.
 - `grid_limit` (_float_): The largest Lagrange multiplier to generate. The grid will contain values distributed between `-grid_limit` and `grid_limit` by default.
 - `grid` (_pandas.DataFrame_): Instead of supplying a size and limit for the grid, users may specify the exact set of Lagrange multipliers they desire using this argument in a DataFrame.
 -  `drop_prot_attr` (_bool_): Flag indicating whether to drop protected attributes from training data.
 - `loss` (_str_): String identifying the loss function for constraints. Options include “_ZeroOne_”, “_Square_”, and “_Absolute_”.
 -  `min_val`: Loss function parameter for “_Square_” and “_Absolute_,” typically the minimum of the range of `y` values.
 -  `max_val`: Loss function parameter for “_Square_” and “_Absolute_,” typically the maximum of the range of `y` values.

Methods:
1. **`fit(dataset)`**: Learns model with less bias.
    ```python
    fit(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_) - Dataset containing true output.
    
    **Returns**: `GridSearchReduction` - Returns self.

2. **`fit_predict`**(_dataset_): Train a model on the input and predict the labels.
    ```python
    fit_predict(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_): Dataset containing true output.
    
    **Returns**:`Dataset` - Transformed dataset.

3. **`fit_transform`**(_dataset_): Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset)
    ```
    **Parameters**:`dataset` (_Dataset_) - Dataset containing true output.
    
    **Returns**:`Dataset` - Transformed dataset.

4. **`predict`**(_dataset_): Obtain the predictions for the provided dataset using the model learned.
    ```python
    predict(dataset)
    ```
    **Parameters**:`dataset` (Dataset) - Dataset containing output values that need to be transformed.
    
    **Returns**:`Dataset` - Transformed dataset.
    
5. **`transform`**(_dataset_): Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**: `dataset` (_Dataset_) - Dataset that needs to be transformed.
    **Returns**: `Dataset` - Transformed dataset.

---
---
## Postprocessing Algorithms in AIF360

1. Calibratyed Equalized Odds
2. Equalized Odds 
3. Reject Option Classification

### 1. Calibrated Equalized Odds
**Class**:`aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing`(_unprivileged_groups, privileged_groups, cost_constraint='weighted', seed=None_)

**Description**: Calibrated equalized odds postprocessing is a post-processing technique that optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective.

**Intialization**:
```python
__init__(unprivileged_groups, privileged_groups, cost_constraint='weighted', seed=None)
```

**Parameters**:
- `unprivileged_groups` (_dict or list(dict)_) – Representation for unprivileged group.
- `privileged_groups` (_dict or list(dict)_) – Representation for privileged group.
 - `cost_contraint` – fpr, fnr or weighted
- `seed` (_int, optional_) – Seed to make predict repeatable.

**Methods**:
- `fit`	
- `fit_predict`	
- `fit_transform`	
- `predict`	
- `transform`	


**Methods Explanation:**
1. **`fit`**:	Compute parameters for equalizing generalized odds using true and predicted scores, while preserving calibration.
    ```python
    fit(dataset_true, dataset_pred)
    ```
    
    **Parameters**:
    - `dataset_true` (_BinaryLabelDataset_) - Dataset containing true labels.
    - `dataset_pred` (_BinaryLabelDataset_) - Dataset containing predicted scores.
    
    **Returns**: `CalibratedEqOddsPostprocessing` – Returns self.
2. **`fit_predict`**:	fit and predict methods sequentially.
    ```python
    fit_predict(dataset_true, dataset_pred, threshold=0.5):
    ```

    **Parameters**:
    - `dataset_true` (_BinaryLabelDataset_) - Dataset containing true labels.
    - `dataset_pred` (_BinaryLabelDataset_) - Dataset containing predicted scores.
    - `threshold` (_float_) - Threshold for converting `scores` to `labels`. 
    Values greater than or equal to this threshold are predicted to be the `favorable_label` (Default is 0.5).
> In `fit_predict` , fit and predict methods sequentially.

3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset_true, dataset_pred)
    ```
    **Parameters**:
    - `dataset_true` (_BinaryLabelDataset_): Dataset containing true labels.
    - `dataset_pred` (_BinaryLabelDataset_): Dataset containing predicted scores.
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.
4. **`predict`**:	Perturb the predicted scores to obtain new labels that satisfy equalized odds - constraints, while preserving calibration.
    ```python
    predict(dataset, threshold=0.5)
    ```
  
    **Parameters**:
    - `dataset` (_BinaryLabelDataset_): Dataset containing scores that needs to be transformed.
    - `threshold` (_float_): Threshold for converting `scores` to `labels`. 
    Values greater than or equal to this threshold are predicted to be the `favorable_label` (Default is 0.5).
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.
5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
   transform(dataset)
    ```
    **Parameters**:`dataset` (_BinaryLabelDataset_) - Dataset that needs to be transformed.
   
    **Returns**:`BinaryLabelDataset` - Transformed dataset.
    
    
### 2. Equalized Odds 

**Class**: `aif360.algorithms.postprocessing.EqOddsPostprocessing`(*unprivileged_groups, privileged_groups, seed=None*)
**Description**: Equalized odds postprocessing is a post-processing technique that solves a linear program to find probabilities with which to change output labels to optimize equalized odds. 
**Initialization**:
``` python
__init__(unprivileged_groups, privileged_groups, seed=None)
```
**Parameters**:
- `unprivileged_groups` (*list(dict)*) – Representation for unprivileged group.
- `privileged_groups` (*list(dict)*) – Representation for privileged group.
- `seed` (*int, optional*) – Seed to make predict repeatable.

**Methods**:
- `fit`	
- `fit_predict`	
- `fit_transform`	
- `predict`	
- `transform`	

**Methods Explanation**:
1. **`fit`**:	Compute parameters for equalizing odds using true and predicted labels.
    ```python
    fit(dataset_true, dataset_pred)
    ```

    **Parameters**:
    - `dataset_true` (*BinaryLabelDataset*): Dataset containing true labels.
    - `dataset_pred` (*BinaryLabelDataset*): Dataset containing predicted labels.
    
    **Returns**: `EqOddsPostprocessing` -  Returns self.
    
2. **`fit_predict`**:	fit and predict methods sequentially.
    ```python
    fit_predict (dataset_true, dataset_pred)
    ```

    **Parameters**: 
    - `dataset_true` (*BinaryLabelDataset*) - Dataset containing true labels.
    - `dataset_pred` (*BinaryLabelDataset*) - Dataset containing predicted labels.
    
    > Note: Run `fit` and `predict` methods sequentially.

3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.

    ```python
    fit_transform(dataset_true, dataset_pred)
    ```
    **Parameters**:
    - `dataset_true` (*BinaryLabelDataset*) - Dataset containing true labels.
    - `dataset_pred` (*BinaryLabelDataset*) - Dataset containing predicted labels.

    **Returns**: `BinaryLabelDataset` - Transformed dataset.
4. **`predict`**:	Perturb the predicted labels to obtain new labels that satisfy equalized odds constraints.

    ```python
    predict(dataset)
    ```
    **Parameters**: `dataset` (*BinaryLabelDataset*) - Dataset containing labels that needs to be transformed.
    
    **Returns**: *BinaryLabelDataset* - Transformed dataset.
5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.

    ```python
    transform(dataset)
    ```
    **Parameters**: `dataset` (*BinaryLabelDataset*) - Dataset that needs to be transformed.
    
    **Returns**: `BinaryLabelDataset` - Transformed dataset.
    
    
### 3. Reject Option Classification 

**Class**:`aif360.algorithms.postprocessing.RejectOptionClassification`(*unprivileged_groups, privileged_groups, low_class_thresh=0.01, high_class_thresh=0.99, num_class_thresh=100, num_ROC_margin=50, metric_name='Statistical parity difference', metric_ub=0.05, metric_lb=-0.05*)

**Description**: Reject option classification is a postprocessing technique that gives favorable outcomes to unpriviliged groups and unfavorable outcomes to priviliged groups in a confidence band around the decision boundary with the highest uncertainty.

**Initialization**:
```python
__init__(unprivileged_groups, privileged_groups, low_class_thresh=0.01, high_class_thresh=0.99, num_class_thresh=100, num_ROC_margin=50, metric_name='Statistical parity difference', metric_ub=0.05, metric_lb=-0.05)
```

**Parameters**:
- `unprivileged_groups` (*dict or list(dict)*) – Representation for unprivileged group.
- `privileged_groups` (*dict or list(dict)*) – Representation for privileged group.
- `low_class_thresh` (*float*) – Smallest classification threshold to use in the optimization. Should be between 0. and 1.
- `high_class_thresh` (*float*) – Highest classification threshold to use in the optimization. Should be between 0. and 1.
- `num_class_thresh` (*int*) – Number of classification thresholds between low_class_thresh and high_class_thresh for the optimization search. Should be > 0.
- `num_ROC_margin` (*int*) – Number of relevant ROC margins to be used in the optimization search. Should be > 0.
- `metric_name` (*str*) – Name of the metric to use for the optimization. Allowed options are “Statistical parity difference”, “Average odds difference”, “Equal opportunity difference”.
- `metric_ub` (*float*) – Upper bound of constraint on the metric value
- `metric_lb` (*float*) – Lower bound of constraint on the metric value

**Methods**:
- `fit`	
- `fit_predict`	
- `fit_transform`	
- `predict`
- `transform`

**Methods Explanation**:
1. **`fit`**:	Estimates the optimal classification threshold and margin for reject option classification that optimizes the metric provided.
    ```python
    fit_predict(dataset_true, dataset_pred)
    ```
    **Parameters**:
    - `dataset_true` (*BinaryLabelDataset*) - Dataset containing the true labels.
    - `dataset_pred` (*BinaryLabelDataset*) - Dataset containing the predicted scores.

    **Returns**:`RejectOptionClassification`: Returns self.
    
    > Note: The `fit` function is a no-op for this algorithm.
    
2. **`fit_predict`**:	fit and predict methods sequentially.
    ```python
    fit_predict(dataset_true, dataset_pred)
    ```
    **Parameters**: 
    - `dataset_true` (*BinaryLabelDataset*): Dataset containing the true labels.
    - `dataset_pred` (*BinaryLabelDataset*): Dataset containing the predicted scores.
    
    > Note: `fit` and `predict` methods sequentially.
    
3. **`fit_transform`**:	Train a model on the input and transform the dataset accordingly.
    ```python
    fit_transform(dataset_true, dataset_pred)
    ```
    **Parameters**:
    - `dataset_true` (*BinaryLabelDataset*) - Dataset containing the true labels.
    - `dataset_pred` (*BinaryLabelDataset*) - Dataset containing the predicted scores.
    
    **Returns**:`BinaryLabelDataset` - Transformed dataset.

4. **`predict`**:	Obtain fair predictions using the ROC method.
    ```python
   predict(dataset)
    ```
    **Parameters**:`dataset` (*BinaryLabelDataset*): Dataset containing scores that will be used to compute predicted labels.
    
    **Returns**:`BinaryLabelDataset`: Output dataset with potentially fair predictions obtained using the ROC method.
5. **`transform`**:	Return a new dataset generated by running this Transformer on the input.
    ```python
    transform(dataset)
    ```
    **Parameters**:`dataset` (*BinaryLabelDataset*) - Dataset containing labels that need to be transformed.
    **Returns**: `BinaryLabelDataset` - Transformed dataset.


---
---






# Acknowledgments

This instructional document incorporates information and data from various sources, including the AIF360 library, AIF360 Github repository, and AIF360 Documentation. We would like to acknowledge and give credit to these sources for their valuable contributions to the content of this document.

- AIF360 Github Repository: [Link](https://github.com/Trusted-AI/AIF360)
- AIF360 Documentation: [Link](https://aif360.readthedocs.io/en/latest/index.html)

We express our gratitude to the creators, contributors, and maintainers of AIF360 for their dedication to fairness and bias mitigation in machine learning. Their resources have been instrumental in the development of this instructional material.
