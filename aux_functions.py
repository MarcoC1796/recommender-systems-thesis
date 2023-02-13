def get_phi_and_Dphi(func, Dfunc, x_k, p_k):
    def phi(alpha):
        return func(x_k + alpha * p_k)

    def Dphi(alpha, is_falttened=True):
        if is_falttened:
            return Dfunc(x_k + alpha * p_k).T @ p_k
        else:
            Dfunc_flattened = Dfunc(x_k + alpha * p_k).flatten
            p_k_flattened = p_k.flatten
