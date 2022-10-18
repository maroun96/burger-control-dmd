import os
import argparse
import numpy as np
import h5py
from pathlib import Path

from utils import load_yml, main_hdf5, func_dict
from burger_env import BurgerEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0,
                    help="seed to generate random control inputs")
    parser.add_argument("-p", "--path", type=str, default='datasets',
                    help="path of directory that contains main hdf5 file")
    args = parser.parse_args()
    seed = args.seed
    dir_name = args.path
    
    np.random.seed(seed=seed)

    cfg = load_yml("config.yml")
    params = cfg["burger_params"]
    main_hdf5(dir_name=dir_name, attr=params)

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
    while True:
        control = np.random.random(2)*(cmax-cmin) + cmin
        X.append(np.copy(u))
        C.append(control)
        u, _, done, _ = burger_env.step(control)
        Y.append(np.copy(u))
        if done:
            break

    with h5py.File(Path(dir_name) / "main.hdf5", 'a') as f:
        in_arr_grp = f["input_arrays"]
        out_arr_grp = f["output_arrays"]
        c_arr_grp = f["control_arrays"]
        in_arr_grp.create_dataset(f'input_array{seed}', data=np.array(X))
        out_arr_grp.create_dataset(f'output_array{seed}', data=np.array(Y))
        c_arr_grp.create_dataset(f'control_array{seed}', data=np.array(C))
        
        

    





   