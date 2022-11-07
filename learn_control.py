import numpy as np

from tqdm import tqdm
from mpc import MPCTracking
from utils import  load_yml, func_dict, H5pyHandler

if __name__ == "__main__":
    iohandler = H5pyHandler(with_control=True)

    Atilde, Btilde, Ur = iohandler.import_model()

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
    iohandler.control_hdf5(attr=mpc_params, C=C)


    
    


    


    