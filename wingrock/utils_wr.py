from typing import Optional, Union, Tuple, List

import torch
from numpy import ndarray
from numpy.core.defchararray import upper
from torch import Tensor, dtype
import torch.nn.functional as F
from abc import ABC, abstractmethod


def clone_required_grad(x: Tensor)-> Tensor:
    """Clone tensor and set requires_grad = True"""
    xnew = torch.tensor(x.detach().cpu().numpy(), device=x.device, requires_grad=True)
    return xnew

def Jacobian(x: Tensor, y: Tensor)->Tensor:
    """Get Jacobian matrix dy/dx based on torch
    Args:
        x : [tensor]
        y : [tensor]
    Returns:
        J : Jacobian matrxi [tensor]
    """

    y = y.view(-1)
    J = torch.zeros((y.numel(), x.numel())).to(x.device)
    for ii in range(y.numel()):
        yy = y[ii]
        dyy_dx = torch.autograd.grad(yy, x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if dyy_dx is None: dyy_dx = torch.zeros_like(x)
        J[ii, :] = dyy_dx.view(1, -1)

    return J


def ToTensor(x: Union[ndarray, List, Tensor], device=None, dtype=torch.float32):
    """to tensor"""
    if x is None:
        return x
    else:
        if isinstance(x, torch.Tensor):
            y = x
        else:
            y = torch.tensor(x)

        if dtype is not None:
            y = y.to(dtype)

        if device is not None:
            return y.to(device)

        return y


class IModel(ABC):

    @abstractmethod
    def fun_input(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            z: GP input (..., nz)
            dzdx: Jacobin of z w.r.t. x (..., nz, nx) or None
        """

        pass

    @abstractmethod
    def fun_tran(self, x: Tensor, c: Tensor, f: Tensor)-> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
            f: GP output (..., nf)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (..., nx, nx) or None
            Af: Jacobin for GP output dF/df (..., nx, nf) or None
        """

        pass

    @abstractmethod
    def fun_meas(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (..., ny, nx) or None
        """

        pass

class IModelEkf(ABC):

    @abstractmethod
    def fun_tran(self, x: Tensor, c: Tensor, w: Tensor)-> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
            w: GP output (..., nf)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (..., nx, nx) or None
            Aw: Jacobin for GP output dF/df (..., nx, nf) or None
        """

        pass

    @abstractmethod
    def fun_meas(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (..., ny, nx) or None
        """

        pass

def chol_add(L0, v, m, id):
    """Compute the updated Cholesky factor after adding rows/columns to the original matrix.
    Args:
        L0: original cholesky factor (n, n)
        v: added column (n, d)
        m: added square block (d, d)
        id: adding index
    Return:
        L: new cholesky factor (n+d, n+d)
    """

    n, d = v.shape
    assert L0.shape == (n, n), "L0 must be square with shape (n, n)"
    assert m.shape == (d, d), "m must be square with shape (d, d)"
    assert 0 <= id <= n, "id must be between 0 and n"

    if id > 0 and id < n:
        """
        original cholesky factor and matrix: L0 = [A0, 0        L0*L0^T = [A0*A0^T, A0*B0^T
                                                   B0, C0]                 B0*A0^T, B0*B0^T + C0*C0^T]
        new cholesky factor and matrix:      L = [A, 0, 0       L*L^T = [A*A^T A*a^T            A*B^T                   = [A0*A0^T  v1    A0*B0^T
                                                  a, b, 0                a*A^T a*a^T + b*b^T    a*B^T + b*c^T              v1^T     m     v2^T
                                                  B, c, C]               B*A^T B*a^T + c*b^T    B*B^T + c*c^T + C*C^T]     B0*A0^T  v2    B0*B0^T + C0*C0^T]
        where: v1 = v[:id, :], v2 = v[id:, :]
        therefore:  A = A0, a^T = A^-1 * v1, B = B0
                    b = chol(m - a*a^T)
                    c^T = b^-1 * (v2^T - a*B^T)
                    C = chol(C0*C0^T - c*c^T)
        """
        # Partition L0
        A0 = L0[:id, :id]  # (id, id)
        B0 = L0[id:, :id]  # (n - id, id)
        C0 = L0[id:, id:]  # (n - id, n - id)

        v1 = v[:id, :]  # (id, d)
        v2 = v[id:, :]  # (n - id, d)

        A, B = A0, B0
        a = torch.linalg.solve_triangular(A0, v1, upper=False).T  # (id, d)
        b = torch.linalg.cholesky(m - a@a.T)  # (d, d)
        c = torch.linalg.solve_triangular(b, v2.T - a @ B0.T, upper=False).T  # (d, n - id)
        C = choldown.chol_downdate(C0, c)

        L = assemble_chol([[A], [a, b], [B, c, C]])

    elif id == 0:
        """
        new cholesky factor: L = [a, 0  , L*L^T = [a*a^T, a*b^T             = [m, v^T
                                  b, C]            b*a^T, b*b^T + C*C^T]       v, L0*L0^T]
        """
        a = torch.linalg.cholesky(m)
        b = torch.linalg.solve_triangular(a, v.T, upper=False).T
        C = choldown.chol_downdate(L0, b)
        L = assemble_chol([a], [b, C])

    elif id == n:
        """
        new cholesky factor: L = [L0, 0  ,      L*L^T = [L0*L0^T,   L0*rho^T                    = [L0*L0^T, v^T
                                  rho, beta]             rho*L0^T   rho*rho^T + beta*beta^T]       v, m]
        """
        rho = torch.solve_triangular(L0, v).T
        beta = torch.linalg.cholesky(m - rho@rho.T)
        L = assemble_chol([[L0], [rho, beta]])

    return L

def assemble_chol(L_dict):
    """
    Args:
        L_dict: [[A], [a, b], [B, c, C], ...]
    Returns:
        L
    """

    def get_n_col(l_dict):
        n_cols = [l_dict[i].shape[1] for i in range(len(l_dict))]
        return sum(n_cols)

    def get_raw(L, n_col_last):
        n_col = get_n_col(L)
        mat_zero = torch.zeros((L[0].shape[0], n_col_last - n_col), dtype=L[0].dtype, device=L[0].device)
        return torch.cat((*L, mat_zero), dim=1)

    n_col_last = get_n_col(L_dict[-1])
    return torch.cat([get_raw(L, n_col_last) for L in L_dict], dim=0)


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def softplus(x):
    return F.softplus(x)

if __name__ == '__main__':
    x = torch.tensor([1.], requires_grad=True)
    x0 = 1*x
    z = 2*x
    y = x0 + z

    print(Jacobian(x, y))
    print(Jacobian(x0, y))