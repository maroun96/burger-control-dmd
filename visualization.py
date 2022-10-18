import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import import_model, import_control, load_yml, func_dict
from burger_env import BurgerEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--mpath", type=str, default='model',
                    help="path of directory that contains the model")
    parser.add_argument("-cp", "--cpath", type=str, default='control',
                    help="path of directory that contains the control sequence")
    
    args = parser.parse_args()
    model_dirname = args.mpath
    control_dirname = args.cpath

    Atilde, Btilde, Ur = import_model(model_dirname)

    A = np.linalg.multi_dot([Ur, Atilde, Ur.T.conj()])
    B = np.linalg.multi_dot([Ur, Btilde])

    cfg = load_yml('config.yml')
    burger_params = cfg["burger_params"]
    initial_func = func_dict.get(burger_params["initial_field"])
    cfunc_list = [lambda x: np.exp(-(15*(x-0.25))**2), lambda x: np.exp(-(15*(x-0.75))**2)]
    burger_env = BurgerEnv(burger_params, initial_func, cfunc_list)
    
    u_init = burger_env.initial_field

    U_num = [u_init]    
    U_dmd = [u_init]

    while True:
        control = np.array([0, 0])
        u, _, done, _ = burger_env.step(control)
        U_num.append(np.copy(u))
        if done:
            break
    
    u = burger_env.reset()
    for _ in range(burger_env.n_steps):
        u = A.dot(u)
        U_dmd.append(u)
    
    # fig, ax = plt.subplots(figsize = (10, 10))

    # cmap = plt.cm.get_cmap('hsv', len(U_dmd))
    # for id, u in enumerate(zip(U_num, U_dmd)):
    #     if id % 200 == 0:
    #         u_num, u_dmd = u
            
    #         ax.plot(burger_env.x, u_num, color=cmap(id), label=f't={id*burger_env.time_step}')
    #         ax.plot(burger_env.x, u_dmd, linestyle = '--', color=cmap(id))
    

    # plt.legend()
    # plt.grid(True, linestyle='--')
    # plt.show()

    # U_num = np.array(U_num)
    # U_dmd = np.array(U_dmd)

    C = import_control(control_dirname)
    solution = [burger_env.initial_field]
    
    burger_env.reset()
    for control in C:
        u, _, _, _ = burger_env.step(control)
        solution.append(np.copy(u))
    
    # fig, ax = plt.subplots(figsize = (10, 10))
    # ax.axhline(y = burger_params["u_ref"], color = 'r', linestyle = '--')

    # for i, u in enumerate(solution):
    #     if i % 200 == 0:
    #         ax.plot(burger_env.x, u, label=f't={i*burger_env.time_step}')
    
    # plt.legend()
    # plt.grid(True, linestyle='--')
    # plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))

    u_ref = burger_params["u_ref"]
    norms = []

    for u in solution:
        norms.append(np.linalg.norm(u-u_ref))
    
    t = np.linspace(0, burger_env.total_time, burger_env.n_steps+1)
    ax.plot(t, norms)
    plt.show()
        



    
        

    



