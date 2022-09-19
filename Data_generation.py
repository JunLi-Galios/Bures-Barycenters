import numpy as np
from numpy import linalg as LA

#     covs = d × d × n array containing covariance matrices
def gen1(covs, sqrt_covs, α, β, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = np.zeros(d,d)
    # evals = np.zeros(d)

    assert rank <= d

    if rank == -1:
        rank = d # full rank

    for i in range(n):
        basis = np.random.rand(d,d)
        # basis[rank:,:] = 0
        evals = np.linspace(α, β, d)
        evals[rank:] = 0

        # r = LA.matrix_rank(basis)
        # assert r == rank

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs


#     covs = d × d × n array containing covariance matrices
def gen2(covs, sqrt_covs, α, β, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = np.zeros(d,d)
    # evals = np.zeros(d)

    assert rank <= d

    if rank == -1:
        rank = d # full rank

    for i in range(n):
        basis = np.random.rand(d,d)
        # basis[rank:,:] = 0
        evals = α + (β - α) * np.random.rand(d)
        evals[rank:] = 0

        # r = LA.matrix_rank(basis)
        # assert r == rank

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs


#     covs = d × d × n array containing covariance matrices
def gen3(covs, sqrt_covs, κ, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = np.zeros(d,d)
    # evals = np.zeros(d)

    assert rank <= d

    if rank == -1:
        rank = d # full rank

    for i in range(n):
        basis = np.random.rand(d,d)
        # basis[rank:,:] = 0
        evals = 1 + (κ-1) * np.random.rand(d)
        if 3*i <= n:
            evals = evals * 0.01
        elif 3*i > 2*n:
            evals = evals*100
        evals[rank:] = 0
        
        # r = LA.matrix_rank(basis)
        # assert r == rank

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs

    
#     covs = d × d × n array containing covariance matrices
def gen4(covs, sqrt_covs, α, β, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = np.zeros(d,d)
    # evals = np.zeros(d)

    assert rank <= d

    if rank == -1:
        rank = d # full rank

    basis = np.random.rand(d,d)
    # basis[rank:,:] = 0
    # r = LA.matrix_rank(basis)
    # assert r == rank

    for i in range(n):
        evals = α + (β - α) * np.random.rand(d)
        evals[rank:] = 0

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs


#     covs = d × d × n array containing covariance matrices
def gen5(covs, sqrt_covs, α, β, mult, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = np.zeros(d,d)
    # evals = np.zeros(d)

    assert rank <= d

    if rank == -1:
        rank = d # full rank

    for i in range(n):
        arr = α + (β-α) * np.random.rand(1+int(d/mult))
        basis = np.random.rand(d,d)
        # basis[rank:,:] = 0
        evals = np.random.choice(arr, d)
        evals[rank:] = 0

        # r = LA.matrix_rank(basis)
        # assert r == rank

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs


def gen6(covs, sqrt_covs, α, β, mult, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # basis = zeros(d,d)
    # evals = zeros(d)
    arr = α + (β-α)*np.random.rand(1+int(d/mult))

    if rank == -1:
        rank = d # full rank

    for i in range(n):
        basis = np.random.rand(d)
        evals = np.random.choice(arr, d)
        evals[rank:] = 0

        covs[:,:,i] = basis @ np.diag(evals) @ basis.T
        sqrt_covs[:,:,i] = basis @ np.diag(evals**0.5) @ basis.T

    return covs, sqrt_covs


def gen7(covs, sqrt_covs, α, β, κ, mult, rank=-1):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    
    a = int(n/6.0)
    b = int(2*n/6.0)
    c = int(3*n/6.0)
    d = int(4*n/6.0)
    e = int(5*n/6.0)
    
    
    c1 = covs[:,:,:a]
    c2 = covs[:,:,a:b]
    c3 = covs[:,:,b:c]
    c4 = covs[:,:,c:d]
    c5 = covs[:,:,d:e]
    c6 = covs[:,:,e:]
    
    sc1 = sqrt_covs[:,:,:a]
    sc2 = sqrt_covs[:,:,a:b]
    sc3 = sqrt_covs[:,:,b:c]
    sc4 = sqrt_covs[:,:,c:d]
    sc5 = sqrt_covs[:,:,d:e]
    sc6 = sqrt_covs[:,:,e:]
    
    covs[:,:,:a], sqrt_covs[:,:,:a] = gen1(c1, sc1, α, β, rank)
    covs[:,:,a:b], sqrt_covs[:,:,a:b] = gen2(c2, sc2, α, β, rank)
    covs[:,:,b:c], sqrt_covs[:,:,b:c] = gen3(c3, sc3, κ, rank)
    covs[:,:,c:d], sqrt_covs[:,:,c:d] = gen4(c4, sc4, α, β, rank)
    covs[:,:,d:e], sqrt_covs[:,:,d:e] = gen5(c5, sc5, α, β, mult, rank)
    covs[:,:,e:], sqrt_covs[:,:,e:] = gen6(c6, sc6, α, β, mult, rank)

    return covs, sqrt_covs
