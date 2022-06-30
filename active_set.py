import numpy as np


def obj_function(input_x: np.ndarray) -> float:
    # Convert to quadratic form
    y = 0.5 * (input_x.T @ G @ input_x) + input_x.T @ c
    return y


if __name__ == '__main__':
    G = np.array([[2, -2],
                  [-2, 4]])
    c = np.array([[-2, -6]]).transpose()
    A = np.array([[-0.5, -1],
                  [1, -2],
                  [1, 0],
                  [0, 1]])
    b = np.array([[-1, -2, 0, 0]]).transpose()

    # initial
    x = np.array([[1, 1/2]]).transpose()

    iter_index = 1
    active_set = np.array([2])

    # check active_set
    if not active_set.size > 0:
        raise Exception("active_set should initial without None to warm up algorithm")

    while True:
        inactive_index = np.delete(np.arange(A.shape[0]), active_set) if active_set.size > 0 else np.arange(A.shape[0])

        A_active = A[active_set, :]
        b_active = b[active_set, :]

        A_inactive = A[inactive_index, :]
        b_inactive = b[inactive_index, :]

        g_k = G @ x + c  # g_k equal to c
        h_k = A_active @ x - b_active

        # Construction of matrix by direct method to find p
        zero_array = np.zeros((len(active_set), len(active_set)))

        top = np.concatenate((G, -A_active.T), axis=1)
        bottom = np.concatenate((A_active, zero_array), axis=1)
        M = np.concatenate((top, bottom), axis=0)

        N = np.concatenate((g_k, h_k), axis=0)

        p_lambda = np.linalg.pinv(M) @ N
        p = -p_lambda[:G.shape[1]]
        lamb = p_lambda[-len(active_set):] if len(active_set) > 0 else np.array([])

        thd = 1e-14

        print(f'-----------{iter_index}-----------')

        if np.all(np.abs(p) <= thd):
            # Calculating Lagrange multipliers
            lamb_new = np.linalg.pinv(A_active.T) @ g_k  # Pseudo inverse can be used for non square matrices
            print(f'lamb_new:{lamb_new.T}')
            if np.any(lamb_new < 0):
                active_set = np.delete(active_set, np.argmin(lamb_new))
                print(f'active_set{active_set}')
            else:
                print(f'active_set{active_set}')
                print(f'======> Find correct answer! \n======> iter_index {iter_index} correct x*: {x.T}')
                break
        else:
            # Calculate alpha_ k. And iteratively update x
            # Conditions other than active set
            print('Calculate alpha_ k and update x')
            mask = A_inactive @ p < 0
            tmp = (b_inactive - A_inactive @ x) / (A_inactive @ p)
            alpha = min(1, np.min(tmp[mask])) if tmp[mask].size > 0 else 1
            if alpha < 1:
                # if exists blocking constrains;
                # Obtain W_k+1 by adding one of the blocking constrains to W_k
                add_index = np.where(tmp == alpha)[0]
                active_set = np.append(active_set, inactive_index[add_index])
            x = x + alpha * p

        iter_index += 1
