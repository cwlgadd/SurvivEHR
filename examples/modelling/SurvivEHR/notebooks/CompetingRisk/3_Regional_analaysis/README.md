# Transfer Learning Experiment

This folder contains an experiment designed to show how a pre-trained
foundation model transfers knowledge across related downstream tasks.

## Background

Foundation models are trained on large and diverse populations. A key
property of such models is their ability to generalise and adapt when
fine-tuned on specific subsets of data. To study this, we compare two
settings:

1. **Pre-trained foundation model**  
   The model is first pre-trained on the full population. It is then
   fine-tuned separately on two sub-populations - patients from two
   different regions of England.  
   - We expect that each fine-tuned model performs best on the
     sub-population it was trained on.  
   - However, because of the prior pre-training, the model should still
     transfer knowledge and perform relatively well when evaluated on
     the *other* sub-population.

3. **Uninitialised (random) model**  
   As a control, we repeat the same fine-tuning experiments starting
   from a model without pre-training.  
   - In this case, we expect a stronger drop in performance when
     evaluated out-of-distribution, since no shared knowledge was
     transferred from pre-training.

This setup allows us to analyse transfer learning behaviour in a simple
and interpretable way.

## Motivation

This experiment is motivated by the idea that pre-training builds a
shared representation of the whole population. Fine-tuning then
specialises the model, but the underlying knowledge helps it adapt
across related tasks. Models trained from scratch lack this shared
representation, making them more brittle when applied out-of-
distribution.

In short, **pre-training enables knowledge transfer and smoother
adaptation across sub-populations**, even when the fine-tuning is
performed on only one subset.

## Experiment Setup

- **Before experiment**: A pre-train the model on the entire population.

For each dataset to be considered:
- **Step 1**: Fine-tune four copies of the model:  
  - *notebook 1*: on the sub-population of London residents, from a randomly initialised model
  - *notebook 2*: on the sub-population of North-Eastern England residents, from a randomly initialised model
  - *notebook 3*: on the sub-population of London residents, from the pre-trained model
  - *notebook 4*: on the sub-population of North-Eastern England residents, from the pre-trained model
- **Step 3**: Evaluate each fine-tuned model on both *all* sub-population
- **Step 4**: Compare with models trained from scratch (no pre-training) (step 1, notebooks 1 and 2), to
	models trained from the pre-trained model (step 1, notebooks 3 and 4)

The results are summarised in a table where **rows indicate
the training population** and **columns indicate the evaluation
population**.

## Results

Results specific to each notebook are shown at the start of their respective notebook. Here we show 
the results alongside each other for comparison.

### Performance

We first present the performance of each model trained on each task. Note that population sizes vary 
and therefore we see a a model trained on a larger population but evaluated out-of-distribution still
performs better than one trained on a smaller population and trained in-distribution. 
We show the results for three survival metrics: Concordance (time-dependent), Integrated Brier Score,
and Negative Bernoulli log-likelihood, respectively.

| Train\Eval (Concordance td) | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | 0.827            | 0.839              |
| North-East                  | 0.754            | 0.748              |
| Pre-trained + London        | 0.84             | 0.849              |
| Pre-trained + North-East    | 0.817            | 0.836              |


| Train\Eval (IBS)            | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | 0.074            | 0.069              |
| North-East                  | 0.081            | 0.078              |
| Pre-trained + London        | 0.072            | 0.067              |
| Pre-trained + North-East    | 0.077            | 0.071              |


| Train\Eval (NBLL)           | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | 0.237            | 0.223              |
| North-East                  | 0.263            | 0.256              |
| Pre-trained + London        | 0.231            | 0.215              |
| Pre-trained + North-East    | 0.25             | 0.232              |



### Performance relative to baseline training scheme

Viewing the results this way it can be hard to disambiguate the relative impact of the distributional
shift. To help, we also present a table with normalised relative difference. For example, when evaluating
the performance (on any dataset)  of a model trained on dataset X, we standardise by the performance 
attained evaluating on dataset X when trained from scratch on the same dataset:

$$
\frac{Eval(Y \mid P, X)}{Eval(X \mid P=0, X)}
$$

where Y is some evaluation sub-population, X is the training sub-population, and P is a binary
indicator to indicate whether we initialised training from the pre-trained model. For concordance
a positive (+) symbol marks improvement, whilst for IBS and INBLL a negative symbol (-) marks
improvement.


| Train X\Eval Y (C_td)       | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **+** 1.43%       |
| North-East                  | **-** 0.71%    | baseline           |
| Pre-trained + London        | **+** 1.57%     | **+** 2.58%       |
| Pre-trained + North-East    | **+** 9.15%    | **+** 11.69%      |


| Train X\Eval Y (IBS)        | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **-** 6.55%        |
| North-East                  | **+** 4.41%      | baseline           |
| Pre-trained + London        | **-** 2.82%      | **-** 10.24%       |
| Pre-trained + North-East    | **-** 0.65%      | **-** 8.68%        |


| Train X\Eval Y (NBLL)       | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **-** 6.02%        |
| North-East                  | **+** 2.89%      | baseline           |
| Pre-trained + London        | **-** 2.67%      | **-** 9.43%        |
| Pre-trained + North-East    | **-** 2.23%      | **-** 9.5%         |

It is clear that there is a marked improvement from pre-training. We now perform a similar scaled comparison, 
but look at the within pre-training scheme differences

$$
\frac{Eval(Y \mid P, X)}{Eval(X \mid P, X)}
$$


| Train X\Eval Y (C_td)       | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **+** 1.43%       |
| North-East                  | **+** 0.71%      | baseline           |
| Pre-trained + London        | baseline         | **+** 1.0%       |
| Pre-trained + North-East    | **-** 2.28%      | baseline      |


| Train X\Eval Y (IBS)        | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **-** 6.55%        |
| North-East                  | **+** 4.41%      | baseline           |
| Pre-trained + London        | baseline         | **-** 7.64%        |
| Pre-trained + North-East    | **+** 8.79%      | baseline           |


| Train X\Eval Y (NBLL)       | London           | North-east England |
|-----------------------------|------------------|--------------------|
| London                      | baseline         | **-** 6.02%        |
| North-East                  | **+** 2.89%      | baseline           |
| Pre-trained + London        | baseline         | **-** 6.95%        |
| Pre-trained + North-East    | **+** 8.04%      | baseline           |

