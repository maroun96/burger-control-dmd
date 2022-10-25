import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI

from burger_env import BurgerEnv
from utils import control_hdf5, func_dict, load_yml, main_hdf5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default='datasets',
                    help="path of directory that contains main hdf5 file")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    seed = rank
    dir_name = args.path
    
    np.random.seed(seed=seed)

    cfg = load_yml("config.yml")
    params = cfg["burger_params"]

    dir_path = Path(dir_name)

    if rank == 0:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    comm.Barrier()

    
    main_hdf5(dir_path=dir_path, attr=params, comm=comm)
    

    func_key = params["initial_field"]
    initial_func = func_dict[func_key]
    cfunc_list = [lambda x: np.exp(-(15*(x-0.25))**2), lambda x: np.exp(-(15*(x-0.75))**2)]
    burger_env = BurgerEnv(params=params,initial_func=initial_func, cfunc_list=cfunc_list)

    cmin = burger_env.action_space.low
    cmax = burger_env.action_space.high

    X = []
    Y = []
    C = []
    
    u = burger_env.reset()
    print(f"Starting --> {rank}")

    while True:
        control = np.random.random(2)*(cmax-cmin) + cmin
        X.append(np.copy(u))
        C.append(control)
        u, _, done, _ = burger_env.step(control)
        Y.append(np.copy(u))
        if done:
            break
    
    print(f"done --> {rank}")

    X = np.array(X)
    Y = np.array(Y)
    C = np.array(C)

    with h5py.File(Path(dir_name) / "main.hdf5", 'a', driver='mpio', comm=comm) as f:
        in_arr_grp = f["input_arrays"]
        out_arr_grp = f["output_arrays"]
        c_arr_grp = f["control_arrays"]
        dset_in = []
        dset_out = []
        dset_c = []
        for i in range(size):
            dset_in.append(in_arr_grp.create_dataset(f'input_array{i}', shape=X.shape, dtype=X.dtype))
            dset_out.append(out_arr_grp.create_dataset(f'output_array{i}', shape=Y.shape, dtype=Y.dtype))
            dset_c.append(c_arr_grp.create_dataset(f'control_array{i}', shape=C.shape, dtype=C.dtype))
        dset_in[rank][:] = X
        dset_out[rank][:] = Y
        dset_c[rank][:] = C
        
        

    





   