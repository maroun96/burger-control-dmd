import os
import numpy as np
import yaml
import h5py
from pathlib import Path
from yaml.loader import SafeLoader

def load_yml(path):
    with open(path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return cfg

def import_model(dir_name: str):
    with h5py.File(Path(dir_name) /'model.hdf5') as f:
        Atilde = f['a_operator_red'][()]
        Btilde = f['b_operator_red'][()]
        Ur = f['pod_modes'][()]
    return Atilde, Btilde, Ur

def import_control(dir_name: str):
    with h5py.File(Path(dir_name) / 'control.hdf5') as f:
        C = f['control_sequence'][()]
    return C 

def merge_data(dataset_path):
    with h5py.File(dataset_path) as f:
        input_arrays = f["input_arrays"].items()
        output_arrays = f["output_arrays"].items()
        control_arrays = f["control_arrays"].items()
        X = []
        Y = []
        C = []

        for x, y, c in zip(input_arrays, output_arrays, control_arrays):
            input_key, input_dataset = x
            output_key, output_dataset = y
            control_key, control_dataset = c
            assert input_key[-1] == output_key[-1] == control_key[-1]

            input_arr = input_dataset[()]
            output_arr = output_dataset[()]
            control_arr = control_dataset[()]

            X.append(input_arr)
            Y.append(output_arr)
            C.append(control_arr)
    
    return np.vstack(X), np.vstack(Y), np.vstack(C)


def main_hdf5(dir_name: str, attr: dict):
    dir_path = Path(dir_name)
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)

    hdf5_path = dir_path / 'main.hdf5'
    
    with h5py.File(hdf5_path, "w") as h5f:
        for key, value in attr.items():
            h5f.attrs[key] = value
        h5f.create_group("input_arrays")
        h5f.create_group("output_arrays")
        h5f.create_group("control_arrays")

def model_hdf5(dir_name: str, attr: dict, A: np.ndarray, B: np.ndarray, Ur: np.ndarray):
    dir_path = Path(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    hdf5_path = dir_path / 'model.hdf5'
    with h5py.File(hdf5_path, "w") as h5f:
        for key, value in attr.items():
            h5f.attrs[key] = value
        h5f.create_dataset("a_operator_red", data=A)
        h5f.create_dataset("b_operator_red", data=B)
        h5f.create_dataset("pod_modes", data= Ur)
    
def control_hdf5(dir_name: str, attr: dict, C: np.ndarray):
    dir_path = Path(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    hdf5_path = dir_path / 'control.hdf5'
    with h5py.File(hdf5_path, "w") as h5f:
        for key, value in attr.items():
            h5f.attrs[key] = value
        h5f.create_dataset("control_sequence", data=C)


func_dict = {
    0: lambda x: np.sin(np.pi*x)
}
