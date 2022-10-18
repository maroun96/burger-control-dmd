import numpy as np

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