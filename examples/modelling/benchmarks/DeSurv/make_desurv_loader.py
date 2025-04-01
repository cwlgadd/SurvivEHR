import numpy as np
import pickle
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def convert_to_desurv(x, t, e, batch_size=256, **kwargs):
    """
    Convert from format produced in ``get_dataloaders`` to the native DeSurv format.
    """
    dataset = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x,torch.float32),
                                                                             (t,torch.float32),
                                                                             (e,torch.long)]])
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

    return data_loader

def convert_to_sklearn_RSF(x, t, e, competing_risk=False, **kwargs):
    """
    Convert from format produced in ``get_dataloaders`` to the sksurv's native RSF format.
    """

    X = pd.DataFrame(x, )    #  columns=list(dm.train_set.tokenizer._stoi.keys())[1:]

    if competing_risk is False:
        y = np.array([(_yk, _yt) for _yk, _yt in zip(e, t)], dtype=[('cens', 'bool'), ('time', '<f8')])
    else:
        # Package does not support Competing Risks
        raise NotImplementedError

    return (X, y)

    
def get_dataloaders(dataset, competing_risk, benchmark, sample_size=None, seed=None, **kwargs):
    """
    Load cross-sectional benchmark data, which was converted from the FastEHR format to be suitable for cross-sectional benchmarks
    """

    # Training samples
    if sample_size is not None:
        save_path =  f"/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_{dataset}/" + f"benchmark_data/N={sample_size}_seed{seed}.pickle" 
    else:
        save_path = f"/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_{dataset}/" + "benchmark_data/all.pickle"
        
    with open(save_path, "rb") as handle:
        print(f"Loading training dataset from {save_path}")
        data_train = pickle.load(handle)
        assert data_train['X_train'].shape[0] == sample_size
    
    # display(data["X_train"].head())
    # display(data["y_train"])
    # print(data.keys())
    
    data = {}
    data["X_train"] = data_train["X_train"]
    data["y_train"] = data_train["y_train"]

    # Test and validation samples

    save_path = f"/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_{dataset}/" + "benchmark_data/all.pickle"
    with open(save_path, "rb") as handle:
        print(f"Loading validation/test datasets from {save_path}")
        data_val_test = pickle.load(handle)
        
    data["X_val"] = data_val_test["X_val"]
    data["y_val"] = data_val_test["y_val"]
    data["X_test"] = data_val_test["X_test"]
    data["y_test"] = data_val_test["y_test"]

    # Convert to correct formats
    x_train = data["X_train"].to_numpy(dtype=np.float32)
    x_val = data["X_val"].to_numpy(dtype=np.float32)
    x_test = data["X_test"].to_numpy(dtype=np.float32)
    
    t_train = np.asarray([i[1] for i in data["y_train"]])
    t_val = np.asarray([i[1] for i in data["y_val"]])        
    t_test = np.asarray([i[1] for i in data["y_test"]])

    if competing_risk is False:
        e_train = np.asarray([0 if i[0] == 0 else 1 for i in data["y_train"]])
        e_val = np.asarray([0 if i[0] == 0 else 1 for i in data["y_val"]])
        e_test = np.asarray([0 if i[0] == 0 else 1 for i in data["y_test"]])
    else:
        e_train = np.asarray([i[0] for i in data["y_train"]])
        e_val = np.asarray([i[0] for i in data["y_val"]])
        e_test = np.asarray([i[0] for i in data["y_test"]])

    # display(x_train.shape)
    # display(type(x_train))
    # display(type(x_train[0,0]))
    # display(e_train.shape)
    # display(type(e_train))
    # display(type(e_train[0]))
    # display(t_train.shape)
    # display(type(t_train))
    # display(type(t_train[0]))
    # print(np.mean(e_train))
    # print(np.mean(t_train))
    # print(np.std(t_train))
    # print(np.mean(x_train))
    # print(t_train.min())
    # print(t_train.max())
    # print(np.unique(e_test, return_counts=True))

    match benchmark.lower():
        case "desurv":
            training_return = convert_to_desurv(x_train, t_train, e_train, **kwargs)
            validation_return = convert_to_desurv(x_val, t_val, e_val, **kwargs)
            testing_return = convert_to_desurv(x_test, t_test, e_test, **kwargs)
        case "sklearn_rsf":
            training_return = convert_to_sklearn_RSF(x_train, t_train, e_train, competing_risk=competing_risk, **kwargs)
            validation_return = convert_to_sklearn_RSF(x_val, t_val, e_val, competing_risk=competing_risk, **kwargs)
            testing_return = convert_to_sklearn_RSF(x_test, t_test, e_test, competing_risk=competing_risk, **kwargs)
        case _:
            raise NotImplementedError(f"`benchmark` must be DeSurv or sklearn_RSF. Got {benchmark}")
    
    return training_return, validation_return, testing_return



