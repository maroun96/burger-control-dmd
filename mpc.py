import numpy as np
import cvxpy as cp

from abc import (
  ABC,
  abstractmethod,
)

class MPC(ABC):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        params: dict
    ):
        self._A = A
        self._B = B
        self._pred_horizon = params["prediction_horizon"]
        self._udim, self._cdim = B.shape
        self._uinit = cp.Parameter(self.udim)
        self._u = cp.Variable((self.udim, self.pred_horizon+1))
        self._c = cp.Variable((self.cdim, self.pred_horizon))

        self._cmin = np.array(params["cmin"])
        self._cmax = np.array(params["cmax"])

        self.prob = None
    
    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._B
    
    @property
    def pred_horizon(self):
        return self._pred_horizon
    
    @property
    def u(self):
        return self._u
    
    @property
    def c(self):
        return self._c
    
    @property
    def udim(self):
        return self._udim
    
    @property
    def cdim(self):
        return self._cdim
    
    @property
    def uinit(self):
        return self._uinit
    
    @property
    def cmin(self):
        return self._cmin
    
    @property
    def cmax(self):
        return self._cmax
    
    @property
    def prob(self):
        return self._prob
    
    @prob.setter
    def prob(self, prob):
        self._prob = prob
    
    @abstractmethod
    def define_prob(self):
        pass

    def mpc_step(self, uinit):
        if not self.prob:
            raise Exception(
                "The optimization problem is not defined. Use the define_prob method to do so"
            ) 
        self.uinit.value = uinit
        self.prob.solve(solver=cp.OSQP, warm_start=True)
        control = self.c[:,0].value
        uinit = self.A.dot(uinit) + self.B.dot(control)

        return control, uinit

class MPCTracking(MPC):
    def __init__(self, A: np.ndarray, B: np.ndarray, params: dict):
        super().__init__(A, B, params)
        self._uref = params["u_ref"]
    
    @property
    def uref(self):
        return self._uref

    def define_prob(self):
        Q = np.eye(self.udim)
        objective = 0
        constraints = [self.u[:,0] == self.uinit]

        for k in range(self.pred_horizon):
            objective += cp.quad_form(self.u[:,k] - self.uref, Q)
            constraints += [self.u[:,k+1] == self.A@self.u[:,k] + self.B@self.c[:,k]]
            constraints += [self.cmin <= self.c[:,k], self.c[:,k] <= self.cmax]
        objective += cp.quad_form(self.u[:,self.pred_horizon] - self.uref, Q)
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

