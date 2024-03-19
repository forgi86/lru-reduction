import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scan_utils import associative_scan, binary_operator_diag
from .reduction import dlyap_torch_direct_diagonal


class LRU(nn.Module):
    def __init__(
        self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, max_phase=math.pi
    ):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im)) # N, U
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im)) # H, N

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features


    def replace_ss_params(self, lambdas, B, C, D):
        lambdas_abs = lambdas.abs()
        self.nu_log = nn.Parameter(torch.log(-torch.log(lambdas_abs)))
        angle = lambdas.angle() % (2*torch.pi) # TODO handle angle=0 (make it 2 pi)
        self.theta_log = nn.Parameter(torch.log(angle))
        gamma_log = torch.log(torch.sqrt(torch.ones_like(lambdas_abs) - torch.square(lambdas_abs)))
        self.gamma_log = nn.Parameter(gamma_log)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        self.B = nn.Parameter(B/gammas)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(D)
        
        self.state_features = lambdas.shape[0]
        self.out_features = C.shape[0]
        self.in_features = B.shape[1]

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        
        lambda_re = lambda_abs * torch.cos(lambda_phase)
        lambda_im = lambda_abs * torch.sin(lambda_phase)
        lambdas = torch.complex(lambda_re, lambda_im)
        #lambdas = lambda_abs*torch.exp(1j*lambda_phase)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B = gammas * self.B
        return lambdas, B, self.C, self.D


    def ss_complex_conjugate_matrices(self, to_numpy=True):
        lambdas, B, self.C, self.D = self.ss_params()

        lambdas_full = torch.zeros(2*self.state_features, device=lambdas.device, dtype=lambdas.dtype)
        lambdas_full[::2] = lambdas
        lambdas_full[1::2] = lambdas.conj()

        # First convert to complex conjugate system....
        A_cc = torch.diag(lambdas_full)
        B_cc = torch.zeros((2*self.state_features, self.in_features), device=lambdas.device, dtype=lambdas.dtype)
        B_cc[::2] = B
        B_cc[1::2] = B.conj()
        C_cc = torch.zeros((self.out_features, 2*self.state_features), device=lambdas.device, dtype=lambdas.dtype)
        C_cc[:, ::2] = 0.5*self.C # we take the real part of the complex conjugate system as output...
        C_cc[:, 1::2] = 0.5*self.C.conj()
        D_real = self.D

        ss_full_params = [A_cc, B_cc, C_cc, D_real]
        if to_numpy:
            ss_full_params = [ss_real_param.detach().numpy() for ss_real_param in ss_full_params]

        return (*ss_full_params, )
    

    def ss_real_matrices(self, to_numpy=True):

        A_full, B_full, C_full, D_full = self.ss_complex_conjugate_matrices(to_numpy=False)
        
        # Then apply transformation to real domain
        T_block = torch.tensor([[1, 1], [1j, -1j]], device=A_full.device, dtype=A_full.dtype)
        T_block_inv = torch.linalg.inv(T_block)
        T_full = torch.block_diag(*([T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([T_block_inv] * self.state_features))

        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_full).real
        C_real = (C_full @ T_full_inv).real
        D_real = D_full 

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [ss_real_param.detach().numpy() for ss_real_param in ss_real_params]

        return (*ss_real_params, )
    

    def hankel_singular_values(self, cc=True):
        lambdas, B, C, _ = self.ss_params()
        
        if cc:
            lambdas = torch.cat([lambdas, lambdas.conj()])
            B = torch.cat((B, B.conj()), dim=0)
            C = torch.cat((C, C.conj()), dim=1)/2
        P = dlyap_torch_direct_diagonal(lambdas, B @ (torch.conj(B)).T)
        Q = dlyap_torch_direct_diagonal(lambdas, (torch.conj(C)).T @ C)
        PQ = (P @ Q)#.to(torch.complex128)
        g = torch.sqrt(torch.linalg.eigvals(PQ)).real
        #g = torch.sqrt(torch.linalg.eigvals(PQ).real) # this gives sometimes numerical errors !?
        #g = torch.sqrt(torch.linalg.eigvals(PQ)).real).to(lambdas.real.dtype)
        return g


    def hankel_singular_values_double(self, cc=True):
        lambdas, B, C, D = self.ss_params()
        
        lambdas = lambdas.to(torch.complex128)
        B = B.to(torch.complex128)
        C = C.to(torch.complex128)

        if cc:
            lambdas = torch.cat([lambdas, lambdas.conj()])
            B = torch.cat((B, B.conj()), dim=0)
            C = torch.cat((C, C.conj()), dim=1)/2

        P = dlyap_torch_direct_diagonal(lambdas, B @ (torch.conj(B)).T)
        Q = dlyap_torch_direct_diagonal(lambdas, (torch.conj(C)).T @ C)
        PQ = (P @ Q)
        g = torch.sqrt(torch.linalg.eigvals(PQ)).real.to(D.dtype)

        return g


    def forward_loop(self, input, state=None):

        # Input size: (B, L, H)
        lambdas, B, C, D = self.ss_params()
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )

        states = []
        for u_step in input.split(1, dim=1): # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = (states @ C.mT).real + input @ D.T

        return output

    @torch.compiler.disable
    def forward_scan(self, input, state=None):

        # Only handles input of size (B, L, H)
        # Batched parallel scan, borrows heavily from https://colab.research.google.com/drive/1RgIv_3WAOW53CS0BnT7_782VKTYis9WG?usp=sharing
        # which in turn borrows from https://github.com/i404788/s5-pytorch
        lambdas, B, C, D = self.ss_params()

        # lambdas is shape (N,) but needs to be repeated to shape (L, N),
        # since input_sequence has shape (B, L, H).
        lambda_elements = lambdas.tile(input.shape[1], 1)
        # Calculate B@u for each step u of each input sequence in the batch.
        # Bu_elements will have shape (B, L, N)
        Bu_elements = input.to(B.dtype) @ B.T
        if state is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + lambdas * state 
        # Vmap the associative scan since Bu_elements is a batch of B sequences.
        # Recall that Lambda_elements has been repeated L times to (L, N),
        # while Bu_seq has shape (B, L, N)
        inner_state_fn = lambda Bu_seq: associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]
        # inner_states will be of shape (B, L, N)
        inner_states = torch.vmap(inner_state_fn)(Bu_elements)

        #y = (inner_states @ self.C.T).real + input_sequences * self.D
        y = (inner_states @ C.T).real + input @ D.T
        return y
    

    def forward(self, input, state=None, mode="scan"):

        if state is None:
            state = torch.view_as_complex(
                torch.zeros((self.state_features, 2), device=input.device)
            ) # default initial state, size N

        match mode:
            case "scan":
                y = self.forward_scan(input, state)
            case "loop":
                y = self.forward_loop(input, state)
        return y


if __name__ == "__main__":

    N = 40 #256 # hidden state dimension
    H = 20 #512 # model dimension (input/output dimension)
    L = 10_000 #2048 # input sequence length
    B = 32 # batch size

    layer = LRU(in_features=H, out_features=H, state_features=N)


    input_sequences = torch.randn((B, L, H)) # multiple sequences
    
    output_sequences = layer(input_sequences)
    output_sequences_scan = layer.forward_scan(input_sequences)

    torch.allclose(output_sequences_scan, output_sequences, 1e-2)