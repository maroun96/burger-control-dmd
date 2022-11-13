import argparse
import mlflow

import numpy as np
import matplotlib.pyplot as plt

from utils import  load_yml, func_dict, H5pyHandler
from burger_env import BurgerEnv

class Visu:
    def __init__(self, iohandler: H5pyHandler, burger_env: BurgerEnv):
        self.iohandler = iohandler
        if self.iohandler.with_control:
            Atilde, Btilde, Ur = self.iohandler.import_model()
            self._B = np.linalg.multi_dot([Ur, Btilde])
            self._C = iohandler.import_control()
        else:
            Atilde, Ur = self.iohandler.import_model()
        self._A = np.linalg.multi_dot([Ur, Atilde, Ur.T.conj()])
        self._burger_env = burger_env
    
    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        self.control_exception()
        return self._B
    
    @property
    def C(self):
        self.control_exception()
        return self._C

    @property
    def burger_env(self):
        return self._burger_env
    
    def unctr_sim(self):
        u_init = self.burger_env.initial_field
        U_num = [u_init]
        self.burger_env.reset()
        while True:
            control = np.array([0, 0])
            u, _, done, _ = self.burger_env.step(control)
            U_num.append(np.copy(u))
            if done:
                break
        return U_num
    
    def unctr_dmd(self):
        u_init = self.burger_env.initial_field
        U_dmd = [u_init]
        u = self.burger_env.reset()
        for _ in range(self.burger_env.n_steps):
            u = self.A.dot(u)
            U_dmd.append(u)
        return U_dmd
    
    def ctr_sim(self, control_seq = None):
        self.control_exception()
        if control_seq is None:
            control_seq = self.C
        u_init = self.burger_env.initial_field
        U_num = [u_init]
        self.burger_env.reset()
        for control in control_seq:
            u, _, _, _ = self.burger_env.step(control)
            U_num.append(np.copy(u))
        return U_num
    
    def ctr_dmd(self, control_seq = None):
        self.control_exception()
        if control_seq is None:
            control_seq = self.C
        u_init = self.burger_env.initial_field
        U_dmd = [u_init]
        u = self.burger_env.reset()
        for control in control_seq:
            u = self.A.dot(u) + self.B.dot(control)
            U_dmd.append(np.copy(u))
        return U_dmd
    
    def plot_unctr(self):
        U_num = self.unctr_sim()
        U_dmd = self.unctr_dmd()
        fig, ax = plt.subplots(figsize = (10, 10))
        plot_lines = []
        fontsize = 15
        cmap = plt.cm.get_cmap('hsv', len(U_dmd))
        for id, u in enumerate(zip(U_num, U_dmd)):
            if id % 200 == 0:
                u_num, u_dmd = u
            
                l1, = ax.plot(self.burger_env.x, u_num, color=cmap(id), label=f't={id*self.burger_env.time_step}')
                l2, = ax.plot(self.burger_env.x, u_dmd, linestyle = '--', color=cmap(id))
                plot_lines.append([l1, l2])

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        fig.suptitle('DMD model vs numerical solution', fontsize=fontsize)
        legend1 = plt.legend(plot_lines[1], ["num", "dmd"], loc=1, fontsize = fontsize)
        plt.legend(loc=2, fontsize = 15)
        plt.gca().add_artist(legend1)
        plt.grid(True, linestyle='--')

        U_num = np.array(U_num)
        U_dmd = np.array(U_dmd)
        mse = (np.square(U_num - U_dmd)).sum(axis=1).mean()

        return fig, mse
    
    def plot_ctr(self, control_seq = None):
        self.control_exception()
        uref = self.burger_env.u_ref
        U_num = self.ctr_sim(control_seq=control_seq)
        U_dmd = self.ctr_dmd(control_seq=control_seq)
        fontsize = 15
        fig1, ax1 = plt.subplots(figsize = (10, 10))
        fig2, ax2 = plt.subplots(figsize = (10, 10))
        ax1.axhline(y=uref, color = 'r', linestyle = '--', label='reference solution')
        ax2.axhline(y=uref, color = 'r', linestyle = '--', label='reference solution')
        for i, u in enumerate(zip(U_num, U_dmd)):
            u_num, u_dmd = u
            if i % 200 == 0:
                ax1.plot(self.burger_env.x, u_num, label=f't={i*self.burger_env.time_step}')
                ax2.plot(self.burger_env.x, u_dmd, label=f't={i*self.burger_env.time_step}')
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)
        ax1.grid(True, linestyle='--')
        ax2.grid(True, linestyle='--')
        fig1.legend(fontsize=fontsize)
        fig2.legend(fontsize=fontsize)
        fig1.suptitle("Optimal control of numerical solution", fontsize=fontsize)
        fig2.suptitle("Optimal control of DMD model", fontsize=fontsize)

        U_num = np.array(U_num)
        U_dmd = np.array(U_dmd)
        tracking_erros_num = np.square(U_num-uref).mean(axis=1)
        tracking_erros_dmd = np.square(U_dmd-uref).mean(axis=1)
                
        return fig1, fig2, tracking_erros_num, tracking_erros_dmd
    
    def plot_opt_ctr(self):
        self.control_exception()
        fig, ax = plt.subplots(figsize = (10, 10))
        fontsize = 15
        c1 = self.C[:, 0]
        c2 = self.C[:, 1]
        t = np.linspace(0, self.burger_env.total_time-self.burger_env.time_step, self.burger_env.n_steps)
        ax.plot(t, c1, label="c1")
        ax.plot(t, c2, label="c2")
        ax.grid(True, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        fig.legend(fontsize=fontsize)
        fig.suptitle("Optimal control sequence", fontsize=fontsize)

        return fig
    
    def control_exception(self):
        if not self.iohandler.with_control:
            raise Exception("Model trained without control")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, 
                    help="Name of the experiment for mlflow tracking")
    parser.add_argument("-id", "--runid", type=str, required=True, 
                    help="Mlflow run id")
    parser.add_argument('--control', default=True, 
                    action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    experiment_name = args.name
    run_id = args.runid
    control_bool = args.control

    mlflow.set_experiment(experiment_name=experiment_name)

    cfg = load_yml("config.yml")
    params = cfg["burger_params"]

    func_key = params["initial_field"]
    initial_func = func_dict[func_key]
    cfunc_list = [lambda x: np.exp(-(15*(x-0.25))**2), lambda x: np.exp(-(15*(x-0.75))**2)]
    burger_env = BurgerEnv(params=params,initial_func=initial_func, cfunc_list=cfunc_list)

    iohandler = H5pyHandler(with_control=control_bool)

    visu = Visu(iohandler=iohandler,burger_env=burger_env)

    with mlflow.start_run(run_id=run_id):
        unctr_fig, mse = visu.plot_unctr()
        mlflow.log_figure(unctr_fig, "num_dmd_unctr.png")
        mlflow.log_metric(key="Mean squared error", value=mse)
        if control_bool:
            ctr_num_fig, ctr_dmd_fig, tracking_errors_num, tracking_errors_dmd = visu.plot_ctr()
            mlflow.log_figure(ctr_num_fig, "ctr_num.png")
            mlflow.log_figure(ctr_dmd_fig, "ctr_dmd.png")
            for i, (error_num, error_dmd) in enumerate(zip(tracking_errors_num, tracking_errors_dmd)):
                mlflow.log_metric(key="Tracking error num", value=error_num, step=i)
                mlflow.log_metric(key="Tracking error dmd", value=error_dmd, step=i)
            opt_ctr_fig = visu.plot_opt_ctr()
            mlflow.log_figure(opt_ctr_fig, "opt_ctr.png")



    
        

    



