# SurvivEHR Experiments

This directory contains all experiments conducted with **SurvivEHR**, organised by chapter:

- **Chapter 1:** Pre-training results
  - 1.1: Training SurvivEHR
  - 1.2: Using SurvivEHR to generate future patient timelines
  - 1.3: Exploring the token concept space of SurvivEHR
  - 1.4: Evaluation of SurvivEHR in capturing risk rankings (via the proposed inter-event concordance metric)
- **Chapter 2:** Fine-tuning results
	- 2.1: Prognostic risk of cardiovascular disease following a Type 2 diabete mellitus diagnosis
		- Zero-shot learning. Can SurvivEHR perform this task without additional task-specific training
		- Fine-tuning. Adapting SurvivEHR to this downstream task
	- 2.2: Prognostic risk of hypertension following a Type 2 diabete mellitus diagnosis
		- Zero-shot learning. Can SurvivEHR perform this task without additional task-specific training
		- Fine-tuning. Adapting SurvivEHR to this downstream task
	- 2.3: Prognostic risk of any future morbidiity following a new diagnosis.
		- Zero-shot learning. Can SurvivEHR perform this task without additional task-specific training
		- Fine-tuning. Adapting SurvivEHR to this downstream task
	- 2.4: Blood pressure experiment. Can SurvivEHR predict future investigation values?
	- 2.5 Combining all results from this section for plotting
- **Chapter 3:** Transfer learning between sub-regions
	- An analysis of performance when fine-tuning and evaluating SurvivEHR on different sub-populations.
		- Can SurvivEHR learn behaviours outside of its training population?
		- Does pre-training on a wdier population improve robustness of downstream tasks trained on
		  smaller sub-populations?
- **Chapter 4:** Model calibration analysis
	- *In progress*
