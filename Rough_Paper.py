import numpy as np
import matplotlib.pyplot as plt

def adeim(initial_basis, F):

    C = np.linalg.pinv(initial_basis) @ F 
            
    R = (initial_basis @ C) - F

    _ , Sv , SrT = np.linalg.svd(R,full_matrices=False)

    Sr = SrT

    CT_inv = np.linalg.pinv(C.T)

    r = window_size

    r = min(r, len(Sv))

    for indx in np.arange(0,r):

        alpha = - R @ Sr[:,indx].reshape(-1,1)
        beta  = CT_inv @ Sr[:,indx].reshape(-1,1)
        del_basis = alpha @ beta.T

        initial_basis = initial_basis + del_basis

    new_basis = initial_basis

    # new_basis , _ = np.linalg.qr(new_basis)

    return new_basis

initial_basis = np.load(r"C:\Users\mohag\Desktop\adeim_check\initial_basis.npy")
q_ref = np.load(r"C:\Users\mohag\Desktop\adeim_check\q_ref.npy")
normalizor = np.load(r"C:\Users\mohag\Desktop\adeim_check\normalizor.npy")

FOM_results = np.load(r"C:\Users\mohag\Desktop\adeim_check\FOM 3000 snapshots Explicit - FD Euler cons.npy")

num_cell = 1500
num_mode = 10
window_size = 10
iter = 11

for iter in np.arange(11,3000):

    future_snapshot = FOM_results[:, :, iter].ravel()

    F = np.zeros((num_cell, num_mode))

    snapshots = np.arange(iter-window_size+1, iter)

    for indx in np.arange(0, window_size-1):

        F[:, indx] = normalizor * (FOM_results[:, :, snapshots[indx]].ravel() - q_ref)

    F[:, -1] = normalizor * (future_snapshot - q_ref)

    new_basis = adeim(initial_basis, F)

    # new_basis , _ ,_ = np.linalg.svd(F)

    FOM_check = new_basis @ new_basis.T @ F[:, -1]

    print(iter)

    plt.clf()
    plt.plot(F[:, -1],linewidth=3 , label='FOM')
    plt.plot(FOM_check,linestyle='--' , label = 'Projected FOM')
    plt.title('iteration: ' + str(iter))
    plt.ylabel('Q cons')
    plt.legend()
    # if iter % 50 == 0 :
    plt.pause(0.001)

    initial_basis = new_basis


plt.show()
    



