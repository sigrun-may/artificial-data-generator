# Data generator for artificial high-throughput data including artificial biomarkers and pseudo-classes.

## Purpose

Since commonly not all true biomarkers and internal dependencies of high-dimensional biological datasets are known with
certainty, artificial data, in contrast, **enable to know the expected outcome in advance**. In artificially generated data, 
it is known in advance which features are relevant and what the distribution of each class is. The remaining 
irrelevant features can be purely random or belong to a pseudo-class. This can be helpful, for example, to make random 
effects transparent. 

In order to develop new methods or to compare existing methods for feature selection (finding biomarkers), 
reference data with known dependencies and importance of the individual features are necessary. 

## Data structure

### Different parts of the data set

A generated artificial data set consists of up to three parts:
1. **Relevant features** belonging to an artificial class: artificial biomarkers 
2. [optional] **Pseudo-classes** (for example a patient's height or gender, which have no association with a particular disease)
3. [optional] **Random data** representing the features (biomarker candidates) that are not associated with any class

This is an example of simulated binary biological data including artificial biomarkers:
![Different blocks of the artificial data.](<./docs/figures/artificial_data.png>)

The number of artificial classes is not limited. Each class is generated individually and then combined with the others.
In order to simulate artificial biomarkers in total, all individual classes have the same number of features in total.

### Data distribution and effect sizes

For each class, either the **normal distribution or the log normal distribution** can be selected. 
The normally distributed data could, for example, represent the range of values of healthy individuals.

In the case of a disease, biological systems are in some way out of balance. 
Extreme changes in values as well as outliers can then be observed.  CITE. 
Therefore, the values of a diseased individual could be simulated with a lognormal distribution.

The different **classes can be shifted** to regulate the effect sizes and to influence the difficulty of data analysis.
Example:
![Different distributions of the classes.](<./docs/figures/distributions.png>)

### Correlations

It is probably likely that biomarkers in healthy individuals usually have a relatively low correlation. On average, 
their values are within a usual "normal" range. In this case, one biomarker tends to be in the upper normal range 
and another biomarker in the lower normal range. In another person, however, it could be exactly the opposite, 
so that the correlation in the healthy individuals would be rather low. Therefore, the **values of healthy people 
could be simulated without any special artificially generated correlations**. 

In the case of a disease, however, a biological system is brought out of balance in a certain way and must react to it. 
For example, this reaction can then happen in a coordinated manner involving several biomarkers, 
or corresponding cascades (e.g. pathways) can be activated or blocked. This can result in a **rather stronger 
correlation of biomarkers in patients suffering from a disease**. To simulate these intra-class correlations, 
a class is divided into a given number of groups with high internal correlation 
(the respective strength can be defined). 

However, a high correlation within a group does not necessarily lead to 
a high correlation to other groups or features of the same class. An example of a class with three 
highly correlated groups but without high correlations between all groups:

![Different distributions of the classes.](<./docs/figures/corr_3_groups.png>)

**Intra-class correlation can be generated for each artificial class**. Any number of groups 
containing correlated features can be combined with any given number of uncorrelated features.

## Pseudo-classes

One option for an element of the generated data set is a pseudo-class. For example, this could be a 
patient's height or gender, which are not related to a specific disease. 

The generated pseudo-class contains the same number of classes with identical distributions as the artificial biomarkers. 
But after the generation of the individual classes, all samples (rows) are randomly shuffled. 
Finally, combining the shuffled data with the original, unshuffled class labels, the pseudo-class no longer 
has a valid association with any class label. Consequently, no element of the pseudo-class should be 
recognized as relevant by a feature selection algorithm.
