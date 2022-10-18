import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from sklearn.preprocessing import PolynomialFeatures

def duffing_oscillator(y, t):
    y1, y2 = y
    dydt1 = y2
    dydt2 = y1 - y1**3 
    dydt = np.array([dydt1, dydt2])
    return dydt

def DMD(X,Xprime,r):
    U,Sigma,VT = np.linalg.svd(X,full_matrices=0) # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T).T).T # Step 2
    Lambda, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(Lambda)
    
    Phi = Xprime @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4
    # alpha1 = Sigmar @ VTr[:,0]
    # b = np.linalg.solve(W @ Lambda,alpha1)
    return Phi, Lambda

def DMDc(X, Xprime, C, p, r):
    Omega = np.concatenate((X, C), axis=0)
    n1 = X.shape[0]
    n2 = C.shape[0]

    U_in, Sigma_in, VT_in = np.linalg.svd(Omega,full_matrices=0)
    U_inr = U_in[:,:p]
    Sigma_inr = Sigma_in[:p]
    VT_inr = VT_in[:p,:]

    U_inr1 = U_inr[:n1,:]
    U_inr2 = U_inr[n1:, :]

    U_out, Sigma_out, VT_out = np.linalg.svd(Xprime, full_matrices=0)
    U_outr = U_out[:,:r]
    Sigma_outr = Sigma_out[:r]
    VT_outr = VT_out[:r,:]

    Atilde = np.linalg.multi_dot([U_outr.T.conj(), Xprime, VT_inr.T,
        np.diag(np.reciprocal(Sigma_inr)), U_inr1.T.conj(), U_outr])
    
    Btilde = np.linalg.multi_dot([U_outr.T.conj(), Xprime, VT_inr.T,
        np.diag(np.reciprocal(Sigma_inr)), U_inr2.T.conj()])
    
    Lambda, W = np.linalg.eig(Atilde)

    Phi = np.linalg.multi_dot([Xprime, VT_inr.T,
        np.diag(np.reciprocal(Sigma_inr)), U_inr1.T.conj(), U_outr, W])

    return Atilde, Btilde, U_outr
    





def collect_data(n_sim, time, n_steps):
    ymin = -1.5
    ymax = 1.5
    X = None
    Xprime = None
    for _ in range(n_sim):
        y = np.random.random(2)*(ymax-ymin) + ymin
        t = np.linspace(0, time, n_steps)
        sol = odeint(duffing_oscillator, y, t)
        if X is None and Xprime is None:
            X = sol.T[:,:-1]
            Xprime = sol.T[:,1:]
        else:
            X = np.concatenate((X, sol.T[:,:-1]), axis=1)
            Xprime = np.concatenate((Xprime, sol.T[:,1:]), axis=1)
    
    return X, Xprime

if __name__ == "__main__":

    n_sim = 300
    time = 15
    n_step = 100

    X, Xp = collect_data(n_sim, time, n_step)

    degree = 5
    poly = PolynomialFeatures(5)
    X_lifted = poly.fit_transform(X.T).T
    Xp_lifted = poly.fit_transform(Xp.T).T

    r = min(X_lifted.shape)
    Phi, Lambda = DMD(X_lifted, Xp_lifted, r)

    ymin = -1.5
    ymax = 1.5
    y = np.random.random(2)*(ymax-ymin) + ymin
    y_lifted = poly.fit_transform(y.reshape((1,-1))).squeeze()

    b = np.linalg.solve(Phi, y_lifted)

    sol_recon = []

    for i in range(n_step):
        x = np.real(Phi@np.linalg.matrix_power(Lambda, i)@b)
        sol_recon.append(x)

    sol_recon = np.array(sol_recon)

    t = np.linspace(0, time, n_step)
    sol_exact = odeint(duffing_oscillator, y, t)

    print(f'Initial condition: {y[0]},   {y[1]}')

    # plt.plot(sol_exact[:, 0], sol_exact[:, 1], label='exact')
    # plt.plot(sol_recon[:, 1], sol_recon[:, 2], label='reconstructed')

    plt.plot(t, sol_exact[:, 0], label='exact')
    plt.plot(t, sol_recon[:, 1], label='reconstructed')

    plt.legend()
    plt.show()
