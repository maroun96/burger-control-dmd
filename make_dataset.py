import argparse

import numpy as np
from mpi4py import MPI

from burger_env import BurgerEnv
from utils import H5pyHandler, func_dict, load_yml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=True, 
                    action=argparse.BooleanOptionalAction)
    parser.add_argument('--append', default=False, 
                    action=argparse.BooleanOptionalAction)
    parser.add_argument('-s','--sampling', required=False, type=str, default="random", 
                    help="sampling strategy", choices=["random", "zero"])
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    seed = rank 
    control_bool = args.control
    sampling_strategy = args.sampling
    append_bool = args.append

    iohandler = H5pyHandler(with_control=control_bool, mpi_comm=comm, append=append_bool)

    if not control_bool:
        print("No control")
    
    np.random.seed(seed=seed)

    cfg = load_yml("config.yml")
    params = cfg["burger_params"]

    if not append_bool:
        iohandler.main_hdf5(attr=params)
    

    func_key = params["initial_field"]
    initial_func = func_dict[func_key]
    cfunc_list = [lambda x: np.exp(-(15*(x-0.25))**2), lambda x: np.exp(-(15*(x-0.75))**2)]
    burger_env = BurgerEnv(params=params,initial_func=initial_func, cfunc_list=cfunc_list)

    cmin = burger_env.action_space.low
    cmax = burger_env.action_space.high
    nc = burger_env.action_shape[0]

    X = []
    Y = []
    if control_bool:
        C = []
    
    u = burger_env.reset()
    print(f"Starting --> {rank}")

    while True:
        X.append(np.copy(u))
        if control_bool:
            if sampling_strategy == "random":
                control = np.random.random(nc)*(cmax-cmin) + cmin
            elif sampling_strategy == "zero":
                control = np.zeros(nc)
            C.append(control)
        else:
            control = np.zeros(nc)
        u, _, done, _ = burger_env.step(control)
        Y.append(np.copy(u))
        if done:
            break
    
    print(f"done --> {rank}")

    X = np.array(X)
    Y = np.array(Y)
    if control_bool:
        C = np.array(C)

    iohandler.append_main(data=X, group_name="input_arrays")
    iohandler.append_main(data=Y, group_name="output_arrays")
    if control_bool:
        iohandler.append_main(data=C, group_name="control_arrays")
        
        

    





   