# Calibration


This folder contains experiemnts that evaluate the calibration of the 
SurvivEHR foundation model. 

## Background


Calibration assesses how well a model’s predicted risks correspond to the actual
observed event frequencies. A perfectly calibrated model assigns survival probabilities
that match empirical outcomes across all risk strata - for example, among patients
predicted to have a 20% 1-year mortality risk, approximately one in five should die 
within that period. Calibration plots provide a visual check of this relationship by 
comparing predicted versus observed event probabilities across quantile bins of model
predictions. In survival settings, calibration is typically evaluated at fixed time 
horizons (e.g., 1-year, 5-year) using Kaplan–Meier or cumulative incidence estimates
within each bin. Well-calibrated models support reliable interpretation of absolute risks,
whereas miscalibration indicates systematic over- or under-prediction across the population.

# Differences to existing literature

Almost all EHR sequence models such as [Delphi-2M](https://doi.org/10.1038/s41586-025-09529-3) 
or MedGPT are trained using a cross-entropy objective to predict the next event, and a regression-
style term for inter-event times. This approach is popular because it is simple: it reuses the 
standard next-token architecture from LLMs, requires no (or minimal) new methodological advances, 
and can be used off-the-shelf. However, this formulation has clear conceptual limitations when 
applied to risk prediction (such as the inability to rank risks in prediction, highlighted in
other experiments). It also leads to further limitations.

Take [Delphi-2M](https://doi.org/10.1038/s41586-025-09529-3) as an example. This paper learns
1) a cross-entropy token classification and
2) an exponential waiting-time loss parameterised by instantaneous rates \( \lambda_j \)

The model assumes that, at each step, multiple exponential clocks run in parallel, one per event type $j$ with rate 
\( \lambda_j \). The waiting time to the next event follows an exponential distribution with rate 
$\lambda^*$. This learns separate rates for each event \( T_j \sim \text{Exp}(\lambda_j) \). The model is able 
to leverage that the *minimum* wait time of multiple exponential clocks is itself exponential with
a rate equal to the sum of all of the rates, giving \( p(T^*) = \lambda^* e^{-\lambda^* T^*} \)

where:
- \( T^* \) is the observed time between events, and  
- \( \lambda^* = \sum_j \lambda_j \) is the total instantaneous rate predicted by the model for each
  event \( j \).

However, in such setups:
1) **Risk over time**
	The instantaenous rate is treated as constant until the next event occurs, implying that risk remains unchanged
	over the prediction window.
	This makes the strong stationarity assumption that your risk of an event does not change over time.
3) **Representation of “time at risk”**
    - Because the model is event-indexed, it never represents time at risk without an event (no full survival curve).
    That is, the model outputs rates for next-event occurrence, not the probability of surviving without events up to a certain time.
    For every input, there is always another event following and so the model never sees examples where no further event occurred.
	As a result, absolute risk (the probability of an event within a fixed time frame) isn’t directly produced.
	To compute an absolute risk, Delphi-2M must apply heuristics.
	- This prevents proper estimation of absolute incidence or survival probability and forces post-hoc
	approximations (e.g. case–control sampling) to compare with real-world rates.
	Consequently, calibration requires heuristic transformations - for example, converting predicted rates
	into “annual incidence” using exponential assumptions and approximating person-time via case–control 
	sampling.
	- Even with the competing-exponential setup, the model on only learns the conditional distribution of which event
	  occurs next, given that _some_ event occurs at $T^*$. It never estimates the marginal waiting-time distribution
	  $p(T_j)$
4) **Prediction horizon**
   No calibration at fixed horizons. Predictions are local (time to next event), not cumulative
   (probability of event within 1 year). It forecasts what the next event will be and how long until it occurs,
   rather than the cumulative risk by a specific future time.
   For instance, Delphi-2M can tell the expected time to the next disease diagnosis, but it doesn’t directly give
   “chance of event X within the next year” as an output.
   Calibrating or evaluating at a fixed horizon (e.g. 1-year risk) is indirect, and must be approximated from the model’s
   predicted rate  \( \lambda_j \) via \( 1- \exp^{-\lambda \times 365 } \)  or simulate the generative process. This is only
   valid if \( \lambda_j \) is approximately stationary over that period which is almost never true in practice.
   

In Extended Data Fig. 3 of the Delphi-2M paper, calibration was estimated indirectly. The model’s instantaneous rates 
$\lambda_j$ were first converted into one-year event probabilities using the exponential assumption 
and observed event frequencies (given by the number of observed events divided by total follow-up person-time) were derived from case–control 
samples that approximated person-time. 
These heuristics were necessary for those reasons outlined above - because the model predicts only the next-event rate, not absolute survival probabilities 
over time. 

# Calculation of calibration

SurvivEHR directly models the full survival function over continuous time through a competing-risks formulation.
This enables principled estimation of absolute event probabilities at fixed horizons (e.g., 1-year or 5-year survival)
without heuristic assumptions about constant hazards or missing person-time. The experiment will therefore construct
calibration plots by grouping patients into quantile bins of predicted survival, estimating observed survival within 
each bin via Kaplan–Meier analysis, and comparing predicted versus observed survival probabilities at pre-defined
time horizons. This provides a robust, interpretable assessment of the model’s reliability in absolute risk prediction
across sub-populations.

## Pre-training calibration

*in-progress*

## Fine-tuning calibration

We define a time horizon of $t^*=5$ years.




