import numpy as np
from aux_functions import get_phi_and_Dphi


def lineSearch(func, Dfunc, x_k, p_k, alpha_max=128, c_1=0.3, c_2=0.6):

    phi, Dphi = get_phi_and_Dphi(func, Dfunc, x_k, p_k)

    phi_0 = phi(0)
    Dphi_0 = Dphi(0)

    def zoom(alpha_lo, alpha_hi):

        i = 1

        # while True
        while i <= 100:
            alpha_j = (alpha_hi + alpha_lo) / 2

            phi_val_j = phi(alpha_j)
            phi_val_lo = phi(alpha_lo)

            if phi_val_j > phi_0 + c_1 * alpha_j * Dphi_0 or phi_val_j >= phi_val_lo:
                alpha_hi = alpha_j
            else:
                Dphi_val_j = Dphi(alpha_j)
                if np.absolute(Dphi_val_j) <= -c_2 * Dphi_0:
                    return alpha_j
                # p. 27
                if Dphi_val_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j

            i += 1

        return alpha_j

    alpha_i_1 = 0
    alpha_i = 1

    i = 1

    while alpha_i <= alpha_max:

        phi_val_i = phi(alpha_i)
        phi_val_i_1 = phi(alpha_i_1)

        if phi_val_i > phi_0 + c_1 * alpha_i * Dphi_0 or (
            phi_val_i >= phi_val_i_1 and i > 1
        ):
            return zoom(alpha_i_1, alpha_i)

        Dphi_val_i = Dphi(alpha_i)
        if np.absolute(Dphi_val_i) <= -c_2 * Dphi_0:
            return alpha_i

        if Dphi_val_i >= 0:
            return zoom(alpha_i, alpha_max)

        alpha_i_1 = alpha_i
        # Multiplicar por 2
        # Alpha_i no pasa alpha_max
        alpha_i = alpha_i * 2

        i += 1

    if alpha_i > alpha_max:
        raise Exception("no alpha found")
