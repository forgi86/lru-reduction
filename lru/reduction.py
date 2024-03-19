import scipy.linalg
import numpy as np
import torch


def dlyap_direct_diagonal(lambdas, Q):
    lhs = np.kron(lambdas, np.conj(lambdas))  # Kronecker product
    lhs = 1 - lhs
    x = Q.flatten()/lhs
    X = np.reshape(x, Q.shape)
    return X

def hankel_singular_values(A, B, C):
    P = scipy.linalg.solve_discrete_lyapunov(A, B @ B.T.conjugate())
    Q = scipy.linalg.solve_discrete_lyapunov(A.T, C.T.conjugate() @ C)
    g = np.sqrt(np.linalg.eigvals(P @ Q))
    return g


def hankel_singular_values_diagonal(lambdas, B, C):
    B = np.matrix(B)
    C = np.matrix(C)
    P = dlyap_direct_diagonal(lambdas, B @ B.H)
    Q = dlyap_direct_diagonal(np.conjugate(lambdas), C.H @ C)
    PQ = P @ Q
    g = np.sqrt(np.linalg.eigvals(PQ).real)
    return g


def hankel_singular_values_cc(lambdas, B, C):
    lambdas_cc, B_cc, C_cc = cc_params(lambdas, B, C)
    g = hankel_singular_values_diagonal(lambdas_cc, B_cc, C_cc)
    return g


def hankel_singular_values_lru(lambdas, B, C, cc=True):
    if cc:
        lambdas, B, C = cc_params(lambdas, B, C)
    return hankel_singular_values_diagonal(lambdas, B, C)
    

def balanced_realization(A, B, C, eps=1e-9):
    P = scipy.linalg.solve_discrete_lyapunov(A, B @ (np.conjugate(B)).T)
    Q = scipy.linalg.solve_discrete_lyapunov(
        (np.conjugate(A)).T, (np.conjugate(C)).T @ C
    )

    T = _balanced_realization_transformation(P, Q, eps=eps)
    T_inv = scipy.linalg.inv(T)

    Ab = T_inv @ A @ T
    Bb = T_inv @ B
    Cb = C @ T

    return Ab, Bb, Cb


def balanced_realization_diagonal(lambdas, B, C, eps=1e-9):
    A = np.matrix(np.diag(lambdas))
    B = np.matrix(B)
    C = np.matrix(C)
    P = dlyap_direct_diagonal(lambdas, B @ B.H)
    Q = dlyap_direct_diagonal(np.conjugate(lambdas), C.H @ C)

    T = _balanced_realization_transformation(P, Q, eps=eps)
    T_inv = scipy.linalg.inv(T)

    Ab = T_inv @ A @ T
    Bb = T_inv @ B
    Cb = C @ T
    return Ab, Bb, Cb


def state_space_truncation(A, B, C, states):
    Ared = A[:states, :states]
    Bred = B[:states, :]
    Cred = C[:, :states]
    return Ared, Bred, Cred


def diagonalization(A, B, C, sort=False):
    lambdas, T = scipy.linalg.eig(A)
    T_inv = np.linalg.inv(T)
    Bd = T_inv @ B
    Cd = C @ T

    if sort:
        lambdas, Bd, Cd = sort_states(lambdas, Bd, Cd)
    return lambdas, Bd, Cd


def complex_realization(lambdas_srt, B_srt, C_srt):

    # Discern cc states from real ones. Is it robust enough?
    is_cc = np.zeros(lambdas_srt.shape, dtype=bool)
    factors = np.zeros(lambdas_srt.shape, dtype=float)*np.nan
    idx = 0
    while idx < lambdas_srt.shape[0]-1:
        factor = np.abs(lambdas_srt[idx] - np.conj(lambdas_srt[idx+1]))/np.abs(lambdas_srt[idx])
        factors[idx] = factor
        if factor < 1e-6:
            is_cc[idx] = True
            is_cc[idx+1] = True
            idx += 2
            continue
        idx += 1

    # For complex-conjugate pairs, keep only the positive pair and add
    # a factor 2 on C
    states_c_plus = (is_cc) & (lambdas_srt.imag > 0)
    lambdas_cp = lambdas_srt[states_c_plus]
    #A_cp = np.diag(lambdas_cp)
    B_cp = B_srt[states_c_plus, :]
    C_cp = 2 * C_srt[:, states_c_plus]

    # For purely real poles, no factor 2 needs to be added
    states_real = ~is_cc
    lambdas_r = lambdas_srt[states_real].real #+1j*1e-10 # innocuous small imaginary part added to handle LRU's parameterization
    #A_r = np.diag(lambdas_r)
    B_r = B_srt[states_real, :]
    C_r = C_srt[:, states_real] # no factor 2 for real poles!

    lambdas_final = np.r_[lambdas_cp, lambdas_r]
    B_final = np.r_[B_cp, B_r]
    C_final = np.c_[C_cp, C_r]

    return lambdas_final, B_final, C_final


def sort_states(lambdas, B, C):
    idx = np.argsort(np.abs(lambdas)+1e-30*(lambdas.imag>0))[::-1] # decreasing magnitude, positive imaginary part first (effective, not so elegant..)
    #idx = np.argsort(1e6*(lambdas_red.imag>0) + np.abs(lambdas_red))[::-1] # imag > 0 part first, decreasing magnitude
    lambdas_srt = lambdas[idx]
    B_srt = B[idx, :]
    C_srt = C[:, idx]
    return lambdas_srt, B_srt, C_srt


# def modal_reduction(lambdas, B, C, D, modes):
#     lambdas_srt, B_srt, C_srt = sort_states(lambdas, B, C)
#     lambdam = lambdas_srt[:modes]
#     Bm = B_srt[:modes, :]
#     Cm = C_srt[:, :modes]
#     Dm =  (C_srt[:, modes:] @ (np.diag(1/(1-lambdam))) @ B_srt[modes: :]).real + D
#     #Dm =  (C_srt[:, modes:]  @ B_srt[modes: :]).real + D
#     return lambdam, Bm, Cm, Dm


def singular_perturbation(A, B, C, D, states):
    A11 = A[:states, :states]
    A12 = A[:states:, states:]
    A21 = A[states:, :states]
    A22 = A[states:, states:]

    B1 = B[:states, :]
    B2 = B[states:, :]
    
    C1 = C[:, :states]
    C2 = C[:, states:]

    tmp = np.linalg.inv(np.eye(A.shape[0] - states) - A22)
    
    Ar = A11 + A12 @ tmp @ A21
    Br = B1 + A12 @ tmp @ B2
    Cr = C1 + C2 @ tmp @ A21
    Dr = (C2 @tmp @ B2).real + D

    return Ar, Br, Cr, Dr


def cc_params(lambdas, B, C, dtype=np.complex128):
    lambdas_cc = np.r_[lambdas, np.conjugate(lambdas)]
    B_cc = np.matrix(np.r_[B, np.conjugate(B)])
    C_cc = 1/2*np.matrix(np.c_[C, np.conjugate(C)])
    return lambdas_cc.astype(dtype), B_cc.astype(dtype), C_cc.astype(dtype)


def lru_reduction_pipeline(lambdas, B, C, D, modes, method="balanced_truncation"):

    match(method):
        case "balanced_truncation":
            Ab, Bb, Cb = balanced_realization_diagonal(lambdas, B, C)
            Ar, Br, Cr = state_space_truncation(Ab, Bb, Cb, modes)
            lambdasf, Bf, Cf = diagonalization(Ar, Br, Cr, sort=True)
            Df = D
        case "balanced_singular_perturbation":
            Ab, Bb, Cb = balanced_realization_diagonal(lambdas, B, C)
            Ar, Br, Cr, Dr = singular_perturbation(Ab, Bb, Cb, D, states=modes)
            lambdasf, Bf, Cf = diagonalization(Ar, Br, Cr, sort=True)
            Df = Dr
        case "balanced_truncation_cc":
            lambdas_cc, B_cc, C_cc = cc_params(lambdas, B, C)
            Ab, Bb, Cb = balanced_realization_diagonal(lambdas_cc, B_cc, C_cc)
            Ar, Br, Cr = state_space_truncation(Ab, Bb, Cb, 2*modes)
            lambdas_srt, B_srt, C_srt = diagonalization(Ar, Br, Cr, sort=True)
            lambdasf, Bf, Cf = complex_realization(lambdas_srt, B_srt, C_srt)
            Df = D
        case "modal_singular_perturbation":
            lambdas_srt, B_srt, C_srt = sort_states(lambdas, B, C)
            lambdasf = lambdas_srt[:modes]
            Bf = B_srt[:modes, :]
            Cf = C_srt[:, :modes]
            #Df =  (C_srt[:, modes:] @ B_srt[modes: :]).real + D
            Df = (C_srt[:, modes:] @ (np.diag(1/(1-lambdas_srt[modes:]))) @ B_srt[modes: :]).real + D
        case "modal_truncation":
            lambdas_srt, B_srt, C_srt = sort_states(lambdas, B, C)
            lambdasf = lambdas_srt[:modes]
            Bf = B_srt[:modes, :]
            Cf = C_srt[:, :modes]
            Df = D

        case _:
            raise ValueError(f"non existing reduction method specified: {method}")
    return lambdasf, Bf, Cf, Df


## Pytorch functions
# useless
# def dlyap_torch_direct(A, Q):
#     lhs = torch.kron(A, A.conj())
#     lhs = torch.eye(lhs.shape[0]) - lhs
#     x = torch.linalg.solve(lhs, Q.flatten())
#     x = np.reshape(x, Q.shape)
#     return x


def _balanced_realization_transformation(P, Q, method="sqrtm", eps=1e-9):

    if method == "sqrtm":
        P_sqrt = scipy.linalg.sqrtm(P + eps*np.eye(P.shape[0]))
        [U, Sd, V] = scipy.linalg.svd(P_sqrt @ Q @ P_sqrt)
        T = P_sqrt @ U @ np.diag(1 / (np.sqrt(np.sqrt(Sd))))
    elif method == "chol":
        Lo = scipy.linalg.cholesky(Q + eps*np.eye(Q.shape[0]), lower=True)
        Lc = scipy.linalg.cholesky(P + eps*np.eye(P.shape[0]), lower=True)
        U, S, VT = np.linalg.svd(Lo.T @ Lc)
        T = Lc @ VT.T @ np.diag(1/np.sqrt(S))
    return T


def dlyap_torch_direct_diagonal(lambdas, Q):
    #d_state = lambdas.shape[0]
    #lhs = lambdas.repeat_interleave(d_state) * lambdas.conj().repeat(d_state)
    lhs = torch.kron(lambdas, lambdas.conj())
    lhs = 1 - lhs
    x = Q.flatten()/lhs
    X = torch.reshape(x, Q.shape)
    return X

def hankel_singular_values_direct_diagonal(lambdas, B, C):
    P = dlyap_torch_direct_diagonal(lambdas, B @ (torch.conj(B)).T)
    Q = dlyap_torch_direct_diagonal(lambdas, (torch.conj(C)).T @ C)
    PQ = P @ Q
    g = torch.sqrt(torch.linalg.eigvals(PQ)).real
    return g