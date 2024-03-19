from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchid.datasets import SubsequenceDataset
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch

DATA_FOLDER = ("F16GVT_Files", "BenchmarkData")
u_var = "Force"
y_var = "Acceleration"

def load_data_dict(datasets="all_train"):
    ds_names =[]
    if (datasets == "ms_train") or (datasets == "all_train"):
            ds_names += [f"F16Data_FullMSine_Level{level}" for level in (1, 3, 5, 7)] # multisine training dataset names

    if (datasets == "sinesw_train") or (datasets == "all_train"):
            ds_names += [f"F16Data_SineSw_Level{level}" for level in (1, 3, 5, 7)] # multisine training dataset names

    #TODO: handle correctly those
    #if (datasets == "specialodd_train") or (datasets == "all_train"):
    #       ds_names += [f"F16Data_SpecialOddMSine_Level{level}" for level in (1, 2, 3)] # multisine training dataset names

    ds_data = []
    ds_names_save = []
    for ds_name in ds_names:
        ds_path = Path(*DATA_FOLDER) / f"{ds_name}.mat"
        ds_mat = sio.loadmat(ds_path)
        if ds_name.startswith("F16Data_SpecialOddMSine_Level"):
            y_all = ds_mat[y_var]
            y_all = np.transpose(y_all, axes=(1, 2, 0)) # R, T*P, C
            u_all = ds_mat[u_var][None, ...]
            u_all = np.transpose(u_all, axes=(1, 2, 0))
            for idx, (y, u) in enumerate(zip(y_all, u_all)):
                ds_names_save.append(f"{ds_name}_rep{idx}")
                ds_data.append((y, u))            
            #y = y.reshape(y.shape[0]*y.shape[1], y.shape[2])
            #u = ds_mat[u_var]
            #u = u.reshape(u.shape[0]*u.shape[1], 1)
        else:
            y = ds_mat[y_var].T
            u = ds_mat[u_var].T
            ds_names_save.append(ds_name)
            ds_data.append((y, u))
        #ds_data.append((y, u))
             
    #ds_data = [sio.loadmat(train_path) for train_path in ds_paths]
    ds_dict = {key: val for key, val in zip(ds_names_save, ds_data)}
    return ds_dict

def make_scalers(dataset):

    # TODO could be written more generically
    scaler_y = StandardScaler()
    scaler_y.fit(dataset[0])

    scaler_u = StandardScaler()
    scaler_u.fit(dataset[1])
    return scaler_y, scaler_u
    
def make_scaled_datasets(ds_dict, scaler_y, scaler_u):

    scaled_datasets = []
    for _, dataset in ds_dict.items():
        y = scaler_y.transform(dataset[0])
        u = scaler_u.transform(dataset[1])
        scaled_datasets.append((y, u))
    return scaled_datasets


def make_subsequence_datasets(scaled_datasets, subseq_len=100):
    subseq_datasets = []
    for (y, u) in scaled_datasets:
        y = torch.from_numpy(y).float()
        u = torch.from_numpy(u).float()
        #print(u.std(), y.std(dim=0))
        subseq_datasets.append(SubsequenceDataset(y, u, subseq_len=subseq_len))
    return ConcatDataset(subseq_datasets)


if __name__ == "__main__":

    ds_dict = load_data_dict()
    scaler_y, scaler_u = make_scalers(ds_dict["F16Data_FullMSine_Level3"])
    scaled_datasets = make_scaled_datasets(ds_dict, scaler_y, scaler_u)
    subsequence_dataset = make_subsequence_datasets(scaled_datasets, subseq_len=100)