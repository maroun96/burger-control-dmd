import torch
import numpy as np
from typing import Callable
from gym import Env
from gym.spaces import Box
from numpy.random import random
from functorch import jacrev


class BurgerEnv(Env):
    def __init__(
        self,
        params: dict,
        initial_func: Callable[[np.ndarray], np.ndarray],
        cfunc_list: list[Callable[[np.ndarray], np.ndarray]],

    ):
        super().__init__()
        self.bounds = params["lower_bound"], params["upper_bound"]
        self.n_grid_points = params["n_grid_points"]
        self._length = self.bounds[1] - self.bounds[0]
        self._dx = self._length/self.n_grid_points
        self._x = np.linspace(self.bounds[0], self.bounds[1], self.n_grid_points)

        self.current_time = 0.0

        self._total_time = params["total_time"]
        self._n_steps = params["n_steps"]
        self._time_step = self._total_time/self._n_steps
        self._viscosity = params["viscosity"]
        self._u_ref = params["u_ref"]

        self._cfl_values = []

        self.observation_shape = self.x.shape
        self.observation_space = Box(
            low=-np.ones(self.observation_shape)*np.inf,
            high=np.ones(self.observation_shape)*np.inf,
            dtype=np.float64
        )

        cmin = np.array(params["cmin"])
        cmax = np.array(params["cmax"])
        assert cmin.shape == cmax.shape

        self.action_shape = cmin.shape
        self.action_space = Box(
            low=cmin,
            high=cmax,
            dtype=np.float64
        )


        self.initial_field = initial_func(self.x)
        self.state = np.copy(self.initial_field)

        assert len(cfunc_list) == cmin.size
        self._field_ext = [func(self.x) for func in cfunc_list]

        self._boundary_condition = params["boundary_condition"]
        assert self.boundary_condition in ["dirichlet", "periodic"]

        if self.boundary_condition == "dirichlet":
            self._n_unknowns = self.n_grid_points - 2
            self._cind = np.arange(1, self.n_grid_points-1)
            self._rind = np.arange(2, self.n_grid_points)
            self._lind = np.arange(0, self.n_grid_points-2)
        elif self.boundary_condition == "periodic":
            self._n_unknowns = self.n_grid_points - 1
            self._cind = np.arange(0, self.n_grid_points-1)
            self._rind = np.arange(1, self.n_grid_points) % (self.n_grid_points - 1)
            self._lind = np.arange(-1, self.n_grid_points-2) % (self.n_grid_points - 1)

        
    @property
    def bounds(self):
        return self._bounds
    
    @property
    def n_grid_points(self):
        return self._n_grid_points
    
    @property
    def state(self):
        return self._state
    
    @property
    def state_field(self):
        return self._state_field
    
    @property
    def total_time(self):
        return self._total_time
        
    @property
    def time_step(self):
        return self._time_step
    
    @property
    def n_steps(self):
        return self._n_steps
    
    @property
    def dx(self):
        return self._dx
    
    @property
    def x(self):
        return self._x
    
    @property
    def field_ext(self):
        return self._field_ext
    
    @property
    def viscosity(self):
        return self._viscosity
    
    @property
    def u_ref(self):
        return self._u_ref
    
    @property
    def current_time(self):
        return self._current_time
    
    @property
    def initial_field(self):
        return self._initial_field
    
    @property
    def boundary_condition(self):
        return self._boundary_condition
    
    @bounds.setter
    def bounds(self, t: tuple[int, int]):
        self._bounds = t
        self._dom_length = t[1] - t[0]
    
    @n_grid_points.setter
    def n_grid_points(self, n):
        self._n_grid_points = n
    
    @state.setter
    def state(self, s):
        self._state = s
    
    @state_field.setter
    def state_field(self, s):
        self._state_field = s
    
    @current_time.setter
    def current_time(self, t):
        self._current_time = t
      
    @initial_field.setter
    def initial_field(self, f):
        self._initial_field = f
           
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.burger_step(action)
        reward = -np.linalg.norm((self.state - self.u_ref))
        self.current_time += self.time_step

        if np.allclose(self._total_time, self.current_time):
            done = True
        else:
            done = False
        
        info = {}

        return self.state, reward, done, info
    
    def rhs_upwind(self, u, control):
        field_list_torch = [torch.from_numpy(f) for f in self.field_ext]
        u_c = u[self._cind]
        u_r = u[self._rind]
        u_l = u[self._lind]
        z = torch.zeros(u_c.shape)
        f = -(torch.maximum(u_c, z)*(u_c-u_l)/self.dx + torch.minimum(u_c, z)*(u_r-u_c)/self.dx) + \
            self.viscosity*(u_r+u_l-2*u_c)/(self.dx**2) + sum([c*f[self._cind] for c,f in zip(control, field_list_torch)])
        return f
    
    def burger_step(self, control):
        n = self._n_unknowns
        I = np.identity(n)
        u = self.state
        u_torch = torch.from_numpy(u)
        control_torch = torch.from_numpy(control)
        J_torch = jacrev(self.rhs_upwind)(u_torch, control_torch)
        if self.boundary_condition == "dirichlet": 
            J = np.delete(J_torch.numpy(), [0,-1], axis=1)
        elif self.boundary_condition == "periodic":
            J = np.delete(J_torch.numpy(), [-1], axis=1)
        f_torch = self.rhs_upwind(u_torch, control_torch)
        f = f_torch.numpy()
        A = I - (self.time_step/2)*J
        u_c = u[self._cind]
        b = A@u_c + self.time_step*f
        u_c = np.linalg.solve(A, b)
        u[self._cind] = u_c
        if self.boundary_condition == "periodic":
            u[-1] = u[0]
        self.state = u
    
    def reset(self):
        self.state = np.copy(self.initial_field)
        self.current_time = 0.0
        return self.state
    
    



    



        
