import argparse
import numpy as np

from tqdm import tqdm
from mpc import MPC, MPCTracking
from utils import import_model, load_yml, control_hdf5, func_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--mpath", type=str, default='model',
                    help="path of directory that contains the model")
    parser.add_argument("-cp", "--cpath", type=str, default='control',
                    help="path of directory that will contain the control sequence")
    args = parser.parse_args()
    model_dirname = args.mpath
    control_dirname = args.cpath

    Atilde, Btilde, Ur = import_model(model_dirname)

    A = np.linalg.multi_dot([Ur, Atilde, Ur.T.conj()])
    B = np.linalg.multi_dot([Ur, Btilde])

    cfg = load_yml("config.yml")
    mpc_params = cfg["mpc_params"]
    burger_params = cfg["burger_params"]

    controller = MPCTracking(A, B, mpc_params)
    controller.define_prob()

    initial_func = func_dict.get(burger_params["initial_field"])
    x = np.linspace(
        burger_params["lower_bound"],
        burger_params["upper_bound"],
        burger_params["n_grid_points"]
    )

    u_init = initial_func(x)
    N = burger_params["n_steps"]
    controls = []

    for _ in tqdm(range(N)):
        control, u_init = controller.mpc_step(u_init)
        controls.append(control)
    
    C = np.array(controls)
    control_hdf5(control_dirname, attr=mpc_params, C=C)


    
    


    


    