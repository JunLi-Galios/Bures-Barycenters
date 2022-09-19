
import numpy as np
import time
from numpy import linalg as LA
from scipy.optimize import minimize

# INPUT:
#     nIter = Number of iterations
#     covs = d × d × n array containing covariance matrices
#     sqrt_covs = d × d × n array containing square roots of matrices in covs
#     objective = length nIter vector to save barycenter objective values in
#     times = length nIter vector to save timings in
#     η = step size
#     distances = length nIter vector to save (W₂)² distance to (sqrt_best)² over training
#     sqrt_best = square root of a d × d matrix that we calculate distances to throughout training (ideally taken to be the true barycenter)
# OUTPUT:
#     square root of d × d matrix that achieves best barycenter functional throughout training
def GD(nIter, covs, sqrt_covs, objective, times, η, distances, sqrt_best):
    start = time.time()
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    # X = np.zeros(d, d)
    X = covs[:,:,0]
    
    # Cache variables for memory efficiency. T refers to transport map
    # T = np.zeros(d,d)
    # evals = np.zeros(d)
    # evecs = np.zeros(d,d)
    
    bestval = np.Inf
    candidate_best = np.zeros(d,d)
    

    for i in range(nIter):
        T = np.zeros(d,d)

        for j in range(n):
            sq = sqrt_covs[:,:,j]

            evals, evecs = LA.eig(sq @ X @ sq)
            objective[i] += np.trace(covs[:,:,j] - 2 * evecs @ np.diag(np.sqrt(evals)) @ evecs.T)

            T = T + sq @ evecs @ np.diag(evals**0.5) @ evecs.T @ sq
        
        
        objective[i] = objective[i]/n + np.trace(X)
        if objective[i] < bestval:
            candidate_best = X
            bestval = objective[i]
        
        
        T = T/n
        distances[i] = bures(sqrt_best, X)
        X = ((1 - η) * np.identity(d) + η * T) @ X @ ((1 - η)* np.identity(d) + η * T)
        times[i] = time.time()-start
    
    return candidate_best**0.5


# INPUT:
#     nIter = Number of iterations
#     covs = d × d × n array containing covariance matrices
#     sqrt_covs = d × d × n array containing square roots of matrices in covs
#     objective = length nIter vector to save barycenter objective values in
#     times = length nIter vector to save timings in
#     η = step size
#     α = lower eigenvalue to threshold at (should be ∼ average minimum eigenvalue of covs)
#     β = upper eigenvalue to threshold at (should be ∼ average maximum eigenvalue of covs)
#     distances = length nIter vector to save (W₂)² distance to (sqrt_best)² over training
#     sqrt_best = square root of a d × d matrix that we calculate distances to throughout training (ideally taken to be the true barycenter)
def EGD(nIter, covs, sqrt_covs, objective, times, η, α, β, distances, sqrt_best):
    start = time.time()
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    
    X = np.covs[:,:,0]
    
    # Cache variables for memory efficiency. T refers to transport map
    # T = zeros(d,d)
    # evals = zeros(d)
    # evecs = zeros(d,d)
    
    
    for i in range(nIter):
        
        T = np.zeros(d,d)

        for j in range(n):
            sq = sqrt_covs[:,:,j]
            
            evals, evecs = LA.eig(sq @ X @ sq)
            objective[i] += np.trace(covs[:,:,j] -2 * evecs @ np.diag(np.sqrt(evals)) @ evecs.T)

            T = T + sq @ evecs @ np.diag(np.sqrt(evals)) @ evecs.T @ sq
        end
    
        objective[i] = objective[i]/n + np.trace(X)
        times[i] = time.time()-start
        distances[i] = bures(sqrt_best, X)
        
        T = T/n
        X = X - η*(np.identity(d) - T)
        
        X = clip(X, α, β)

    return X

# INPUT
#     covs = d × d × n array containing covariance matrices
#     sqrt_covs = d × d × n array containing square roots of matrices in covs
#     X = starting covariance matrix 
#     objective = length nIter vector to save barycenter objective values in
#     times = length nIter vector to save timings in
#     ηs = array of length n of stepsizes, or single number (in which case that step size is used for all steps)
#     sqrt_bary = square root of true barycenter
def SGD(covs, sqrt_covs, X, objective, times, ηs, sqrt_bary=None):
    start = time.time()
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    if sqrt_bary is None:
        sqrt_bary = np.identity(d)
    
    
    # Cache variables for memory efficiency. T refers to transport map
    # T = np.zeros(d,d)
    # evals = np.zeros(d)
    # evecs = np.zeros(d,d)

    for i in range(n):
        if len(ηs) == 1:
            η = ηs[0] 
        else:
            η = ηs[i]
        
        sq = sqrt_covs[:,:,i]
        evals, evecs = LA.eig(sq @ X @ sq)
        T = sq @ evecs @ np.diag(evals**(-0.5)) @ evecs.T @ sq
        times[i] = time.time()-start
        objective[i] = bures(sqrt_bary, X)
        X = ((1 - η)*np.identity(d) + η*T) @ X @ ((1-η)*np.identity(d) + η * T)

    return objective, times



# INPUT
#     covs = d × d × n array containing covariance matrices
#     sqrt_covs = d × d × n array containing square roots of matrices in covs
#     objective = length nIter vector to save barycenter objective values in
#     times = length nIter vector to save timings in
#     ηs = array of length n of stepsizes, or single number (in which case that step size is used for all steps)
#     α = lower eigenvalue to threshold at (should be ∼ average minimum eigenvalue of covs)
#     β = upper eigenvalue to threshold at (should be ∼ average maximum eigenvalue of covs)
def ESGD(covs, sqrt_covs, objective, times, ηs, α, β):
    start = time.time()
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    
    # X = np.zeros(d, d)
    X = covs[:,:,0]
    
    # Cache variables for memory efficiency. T refers to transport map
    # T = zeros(d,d)
    # evals = zeros(d)
    # evecs = zeros(d,d)
    
    
    for i in range(n):
        if len(ηs) == 1:
            η = ηs[0] 
        else:
            η = ηs[i]
        
        sq = sqrt_covs[:,:,i]
        evals, evecs = LA.eig(sq @ X @ sq)
        T = sq @ evecs @ np.diag(evals ** (-0.5)) @ evecs.T @ sq
        times[i] = time.time()-start
        objective[i] = bures(np.identity(d), X)
        X = X - η*(np.identity(d) - T)
        
        X = clip(X, α, β)

    return objective, times


# INPUT
#     covs = d × d × n array containing covariance matrices
#     verbose = boolean indicating whether SDP solver should be verbose
#     maxIter = maximum number of iterations
# OUTPUT
#     the barycenter (d × d matrix)
def SDP(covs, verbose=False, maxIter=5000):
    d = np.shape(covs)[0]
    n = np.shape(covs)[2]
    
    Σ = Variable(d,d)
    Ss = [Variable(d,d) for _ in 1:n]
    cons = ({'type': 'eq', 'fun':})
    constr = [([covs[:,:,i] Ss[i]; Ss[i]' Σ] ⪰ 0) for i in 1:n]
    fun = tr(Σ) - 2*mean(tr.(Ss))
    minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    problem.constraints += constr
    problem.constraints += (Σ ⪰ 0)
    optimizer = SCS.Optimizer(verbose = verbose)
    MOI.set(optimizer, MOI.RawParameter("max_iters"), maxIter)
    solve!(problem, optimizer)
    return Σ.value
end


# Function that clips eigenvalues of X to specified range, in place
def clip(X, α, β):
    evals, evecs = LA.eig(X)
    X = evecs @ np.diag(np.clip(evals, α, β)) @ evecs.T
    return X


# Calculates (W₂)² between (sq)^2 and x
def bures(sq,x):
    evals, evecs = LA.eig(sq @ x @ sq)

    return np.trace(x + sq @ sq - 2 * evecs @ np.diag(e.values**0.5)) @ evecs.T)

# Calculates barycenter functional of X over the dataset [sqrt_covs[:,:,i]² for i in 1:n]
def barycenter_functional(sqrt_covs, X)
    arr = [bures(sqrt_covs[:,:,i], X) for i in range(np.shape(sqrt_covs)[2])]
    return np.mean(arr)
