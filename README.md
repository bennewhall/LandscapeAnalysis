# LandscapeAnalysis
## Benjamin Newhall 3/5/21

LandscapeAnalysis is example code on how to use Landscape analysis tools with pyscf.

Ben Newhall

## Installation

For landscape analysis tools, LandscapeAnalysis uses a python port of the popular R package [flacco](https://github.com/kerschke/flacco/blob/master/README.md)

To install `flacco`, make sure you first have R installed then run

```R
install.packages("flacco", dependencies = TRUE)
```
inside the R interpereter

After flacco is installed, install the python port `pflacco` with 

```
pip install pflacco
```

## Basic Usage

The goal of landscape analysis is to use statistical methods to determine the characteristics of a landscape based off of it's *features*.

These features are computed by running different calculations on a random sample from the landscape.

Here is a step by step example on how to calculate the Features of a given landscape.

```python
    #Landscape Analysis
    sample = create_initial_sample(100, m, type = 'random', lower_bound=lower_bound, upper_bound=upper_bound)
    obj_val = []

    
    for s in sample:
      obj_val.append(float(E_hf(s)))
    
   
    feat_obj = create_feature_object(sample, obj_val,minimize = True, lower=-10.0, upper=10.0)

    ela_features = calculate_feature_set(feat_obj,"ela_meta")

```

First create a sample.

```python
    sample = create_initial_sample(100, m, type = 'random', lower_bound=lower_bound, upper_bound=upper_bound)
```
This creates an array of 100 `random` parameter array's, each of size `m`, where each element is between `lower_bound` and `upper_bound`


Next calculate the values of the objective function for each element in the sample array.

```python
    obj_val = []

    
    for s in sample:
      obj_val.append(float(E_hf(s)))

```
E_hf is just the objective function I am using in this example, but it really could be anything.

Then create a flacco *feature object* that stores the data relating to this landscape analysis.

```python
feat_obj = create_feature_object(sample, obj_val,minimize = True, lower=-10.0, upper=10.0)
```

To run the analysis, do :

```python

    ela_features = calculate_feature_set(feat_obj,"ela_meta")
```
Where `ela_features` is a python dict mapping feature -> value and `ela_meta` is one of many different feature sets that flacco supports.

For more information on feature sets look [here](http://kerschke.github.io/flacco-tutorial/site/quickstart/)

For a good explanation of many features and a great overall review of landscape analysis and its uses look at this [survey paper](https://www.researchgate.net/publication/275652955_Algorithm_selection_for_black-box_continuous_optimization_problems_A_survey_on_methods_and_challenges)

## Examples
`read_qm7_analyze.py` - shows how to perform landscape analysis on the rhf landscapes on the qm7 dataset. The output is stored in a pandas dataframe.

`gen_arbH_surf.py` - shows how to perform landscape analysis on datasets of the form described in `readwriteH.py`


## Future Steps

* Creating a pyscf module for landscape analysis
* parallelizing landscape analysis
* Using computed features to train a meta model to describe performance with pyscf solvers
