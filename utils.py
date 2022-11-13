import os
import glob
import numpy as np
import yaml
import h5py
from pathlib import Path
from yaml.loader import SafeLoader

class H5pyHandler:
    main_dirname = "datasets"
    model_dirname = "model"
    control_dirname = "control"

    def __init__(self, with_control: bool, mpi_comm = None, append = False):
        self.with_control = with_control
        self.mpi_comm = mpi_comm
        self.append = append

        if self.append:
            self.current_idx = self._get_data_idx() + 1
        else:
            self.current_idx = 0
    
    def main_hdf5(self, attr: dict):
        assert not self.append
        rank = self.mpi_comm.Get_rank()
        dir_path = Path(os.path.dirname(__file__)) / self.main_dirname
        if rank == 0:
            self.makedir(dir_path)
        self.mpi_comm.Barrier()

        with h5py.File(dir_path / 'main.hdf5', "w", driver='mpio', comm=self.mpi_comm) as h5f:
            for key, value in attr.items():
                h5f.attrs[key] = value
            h5f.create_group("input_arrays")
            h5f.create_group("output_arrays")
            if self.with_control:
                h5f.create_group("control_arrays")
    
    def append_main(self, data: np.ndarray, group_name: str):
        size = self.mpi_comm.Get_size()
        rank = self.mpi_comm.Get_rank()
        self.assert_grpname(group_name)

        dir_path = Path(os.path.dirname(__file__)) / self.main_dirname
        with h5py.File(dir_path / "main.hdf5", 'a', driver='mpio', comm=self.mpi_comm) as f:
            dset = []
            grp = f[group_name]
            for i in range(self.current_idx, size + self.current_idx):
                dset.append(grp.create_dataset(f'{group_name[:-1]}{i}', shape=data.shape, dtype=data.dtype))
            dset[rank][:] = data
    
    def merge_data(self, group_name: str):
        self.assert_grpname(group_name)
        dir_path = Path(os.path.dirname(__file__)) / self.main_dirname
        with h5py.File(dir_path / "main.hdf5", 'r') as f:
            arrays = f[group_name].values()
            data = []
            for arr in arrays:
                data.append(arr[:])
        return np.vstack(data)

    def model_hdf5(self, attr: dict, A: np.ndarray, Ur: np.ndarray, B: np.ndarray = None):
        if B is not None:
            assert self.with_control
        dir_path = Path(os.path.dirname(__file__)) / self.model_dirname
        self.makedir(dir_path)

        with h5py.File(dir_path / 'model.hdf5', "w") as f:
            for key, value in attr.items():
                f.attrs[key] = value
            f.create_dataset("a_operator_red", data=A)
            f.create_dataset("pod_modes", data= Ur)
            if self.with_control:
                f.create_dataset("b_operator_red", data=B)
    
    def control_hdf5(self, attr: dict, C: np.ndarray):
        dir_path = Path(os.path.dirname(__file__)) / self.control_dirname
        self.makedir(dir_path)
        with h5py.File(dir_path / 'control.hdf5', "w") as f:
            for key, value in attr.items():
                f.attrs[key] = value
            f.create_dataset("control_sequence", data=C)
    
    def import_model(self):
        dir_path = Path(os.path.dirname(__file__)) / self.model_dirname
        with h5py.File(dir_path /'model.hdf5', 'r') as f:
            Atilde = f['a_operator_red'][:]
            if self.with_control:
                Btilde = f['b_operator_red'][:]
            Ur = f['pod_modes'][:]
        if self.with_control:
            return Atilde, Btilde, Ur
        else:
            return Atilde, Ur
    
    def import_control(self):
        dir_path = Path(os.path.dirname(__file__)) / self.control_dirname
        with h5py.File(dir_path/ 'control.hdf5', 'r') as f:
            C = f['control_sequence'][:]
        return C

    def _get_data_idx(self):
        dir_path = Path(os.path.dirname(__file__)) / self.main_dirname
        with h5py.File(dir_path / "main.hdf5", 'r') as f:
            input_arrays = f["input_arrays"]
            last_array_key = list(input_arrays.keys())[-1][-1]
        return int(last_array_key)


    def assert_grpname(self, group_name):
        assert group_name in ["input_arrays", "output_arrays", "control_arrays"]
        if group_name == "control_arrays":
            assert self.with_control

    @staticmethod
    def makedir(dir_path: Path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
def load_yml(path):
    with open(path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return cfg

def import_model(dir_name: str):
    with h5py.File(Path(dir_name) /'model.hdf5', 'r') as f:
        Atilde = f['a_operator_red'][()]
        Btilde = f['b_operator_red'][()]
        Ur = f['pod_modes'][()]
    return Atilde, Btilde, Ur

def import_control(dir_name: str):
    with h5py.File(Path(dir_name) / 'control.hdf5', 'r') as f:
        C = f['control_sequence'][()]
    return C 

def merge_data(dataset_path, control=True):
    with h5py.File(dataset_path, 'r') as f:
        input_arrays = f["input_arrays"].items()
        output_arrays = f["output_arrays"].items()
        if control:
            control_arrays = f["control_arrays"].items()
            arrays_list = [input_arrays, output_arrays, control_arrays]
        else:
            arrays_list = [input_arrays, output_arrays]
        X = []
        Y = []
        if control:
            C = []

        for t in zip(*arrays_list):
            input_key, input_dataset = t[0]
            output_key, output_dataset = t[1]
            if control:
                control_key, control_dataset = t[2]
            assert input_key[-1] == output_key[-1]

            if control:
                output_key[-1] == control_key[-1]

            input_arr = input_dataset[()]
            output_arr = output_dataset[()]
            if control:
                control_arr = control_dataset[()]

            X.append(input_arr)
            Y.append(output_arr)

            if control:
                C.append(control_arr)
    if control:
        return np.vstack(X), np.vstack(Y), np.vstack(C)
    else:
        return np.vstack(X), np.vstack(Y)

def main_hdf5(dir_path: Path, attr: dict, comm, control = True):
    hdf5_path = dir_path / 'main.hdf5'
    
    with h5py.File(hdf5_path, "w", driver='mpio', comm=comm) as h5f:
        for key, value in attr.items():
            h5f.attrs[key] = value
        h5f.create_group("input_arrays")
        h5f.create_group("output_arrays")
        if control:
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

def clear_slurm():
    if not os.path.exists("slurm"):
        os.makedirs("slurm")
    files = glob.glob("slurm/*")
    for f in files:
        os.remove(f)


func_dict = {
    0: lambda x: np.sin(np.pi*x)
}
