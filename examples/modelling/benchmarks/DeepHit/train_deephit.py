import numpy as np
import torchtuples as tt # Some useful functions
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from scipy.integrate import trapz
import math
import matplotlib.pyplot as plt

def get_tokens_for_stratification(dm, custom_outcomes_method):
    """
    If we want to stratify RMST or (TODO) metrics, but number of existing conditions, this method gets the 
    """
    # Get the indicies for the diagnoses used to stratify patient groups (under the SurvivEHR setup)
    conditions = custom_outcomes_method(dm)
    encoded_conditions = dm.tokenizer.encode(conditions)                    # The indicies of the MM events in the xsectional dataset (not adjusted for UNK/PAD/static data)
    
    # Get the number of baseline static variables (after one-hot encoding etc), and the vocab size excluding PAD and UNK tokens
    num_cov = dm.train_set[0]["static_covariates"].shape[0]
    num_context_tokens = dm.tokenizer._event_counts.shape[0] - 1            # Removing UNK token, which is not included in xsectional datasets
    
    # Convert the `encoded_conditions` indicies to the equivalent in the xsectional dataset
    encoded_conditions_xsec = [_ind + num_cov - 1 for _ind in encoded_conditions]

    return encoded_conditions_xsec, (num_cov, num_context_tokens)

def run_experiment(dataset_train,
                   dataset_val,
                   dataset_test, 
                   meta_information,
                   num_nodes = [32, 32],
                   batch_norm = True,
                   dropout = 0.1,
                   epochs = 100,
                   batch_size = 256,
                   bins = 200,                       # Default of 10: does very poorly. Increasing, improving results.
                   dm = None,
                   custom_outcomes_method = None,
                   learning_rate = None,
                  ):

    num_xsectional_in_dims = dataset_train[0].shape[1]

    # If given (for RMST stratification), get the outcomes to stratify counts of, plus do some additional sense checking.
    if dm is not None and custom_outcomes_method is not None:
        # Outcome tokens to stratify on
        encoded_conditions_xsec, (num_cov, num_context_tokens) = get_tokens_for_stratification(dm, custom_outcomes_method)

        # Ensure provided datasets align with original FastEHR data
        assert num_xsectional_in_dims == num_cov + num_context_tokens, f"{num_xsectional_in_dims} != {num_cov} + {num_context_tokens}"
        assert dm.tokenizer._event_counts.shape[0] + num_cov -1 == num_xsectional_in_dims
    
    # t_eval = np.linspace(0, 1, 1000)                           # the time grid which we generate over
    time_grid = np.linspace(start=0, stop=1 , num=300)           # the time grid which we calculate scores over
            
    # Train with off-the-shelf parameter setup
    net = tt.practical.MLPVanilla(num_xsectional_in_dims, num_nodes, bins, batch_norm, dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=meta_information["cuts"])

    # Learning rate
    if learning_rate is None:
        lr_finder = model.lr_finder(dataset_train[0], dataset_train[1], batch_size, tolerance=3)
        print(f"lr_finder best lr: {lr_finder.get_best_lr()}")
        # Documentation states this over-estimates the best LR, so set it slightly smaller (as per example), but keep it within reasonable bounds
        lr_exponent = math.floor(math.log10(lr_finder.get_best_lr()))
        lr_exponent = min(max(lr_exponent, -3), -2)
        print(f"setting to lr: {10**lr_exponent}")
        model.optimizer.set_lr(10 ** lr_exponent)
    else:
        model.optimizer.set_lr(learning_rate)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(dataset_train[0], dataset_train[1], batch_size, epochs, callbacks, val_data=dataset_val)
    
    surv = model.predict_surv_df(dataset_test[0])

    if False:
        _ = log.plot()
        plt.savefig(f"figs/loss.png")
    
        surv.iloc[:, :5].plot(drawstyle='steps-post')
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')
        plt.savefig(f"figs/step.png")
        
        surv = model.interpolate(bins).predict_surv_df(dataset_test[0])
        
        surv.iloc[:, :5].plot(drawstyle='steps-post')
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')
        plt.savefig(f"figs/km.png")


    ###########################
    # Get metrics             #
    ###########################   
    ev = EvalSurv(surv, dataset_test[1][0], dataset_test[1][1], censor_surv='km')
    ctd = ev.concordance_td()
    ibs = ev.integrated_brier_score(time_grid)
    inbll= ev.integrated_nbll(time_grid)
    returns = {"ctd": ctd,
               "ibs": ibs,
               "inbll": inbll,
              }
    
    ###########################
    # Get RMST Survival times #
    ###########################      
    if dm is not None and custom_outcomes_method is not None:

        obs_RMST_by_number_of_preexisting_conditions = [[] for _ in range(len(encoded_conditions_xsec))]
        pred_RMST_by_number_of_preexisting_conditions = [[] for _ in range(len(encoded_conditions_xsec))]
        for sample in range(surv.shape[1]):
            # Get the number of pre-existing conditions
            sample_stratification_label = np.sum(dataset_test[0][sample][encoded_conditions_xsec] == 1)
            # Get the RMST predicted under the survival curve
            samples_below_cutoff = surv.index.to_numpy() < 1        # as with other methods, we restrict up until standardised time 1
            # 
            sample_predicted_rmst = trapz(surv[sample].to_numpy()[samples_below_cutoff], surv.index.to_numpy()[samples_below_cutoff])
            pred_RMST_by_number_of_preexisting_conditions[sample_stratification_label].append(sample_predicted_rmst)
            
            if dataset_test[1][1][sample] != 0:
                # Get the observed RMST - warning: this is IGNORING CENSORING
                t_obs = dataset_test[1][0][sample]
                if t_obs <= 1:
                    # t_obs = np.max((t_obs, 1))
                    obs_RMST_by_number_of_preexisting_conditions[sample_stratification_label].append(t_obs)

        # Average across samples of RMST within each stratification
        pred_RMST = [np.mean(_strat) if len(_strat) > 0 else np.nan for _strat in pred_RMST_by_number_of_preexisting_conditions]
        approx_obs_RMST = [np.mean(_strat) if len(_strat) > 0 else np.nan for _strat in obs_RMST_by_number_of_preexisting_conditions]

        returns = {**returns,
                   "pred_RMST": pred_RMST,
                   "approx_obs_RMST": approx_obs_RMST,
                  }
    
    return returns
    

    # model_names.append(model_name)
    # all_ctd.append(ctd)
    # all_ibs.append(ibs)
    # all_inbll.append(inbll)
    # all_pred_RMST.append(pred_RMST_by_number_of_preexisting_conditions)
    # all_obs_RMST.append(obs_RMST_by_number_of_preexisting_conditions)
