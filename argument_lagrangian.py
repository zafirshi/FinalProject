import numpy as np


def obj_function(G: np.ndarray, c: np.ndarray, x: np.ndarray) -> float:
    """
    calculate object function value
    :param G:  SPD matrix of objective function
    :param c:  First order term matrix of objective function
    :param x:  Argument vector(column)
    :return: Function value
    """
    y = 0.5 * (x.T @ G @ x) + x.T @ c
    return y.item()


def grad_obj_function(G: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    calculate gradient
    using the numerical form of quadratic objective function

    :param G:  SPD matrix of objective function
    :param c:  First order term matrix of objective function
    :param x:  Argument vector(column)
    :return: Gradient vector(column)
    :rtype: np.ndarray
    """
    y = G @ x + c
    return y


def aug_obj(input_x, lamb, mu):
    c_x = A @ input_x - b
    # update
    c_x = np.where(c_x > 0, 0, -c_x)
    return obj_function(G, c, input_x) - lamb.T @ c_x + 0.5 * mu * (c_x.T @ c_x)


def aug_grad(input_x, lamb, mu):
    # update
    tmp = A @ input_x - b
    c_x = np.where(tmp > 0, 0, -tmp)
    lamb_new = np.where(c_x > 0, lamb, 0)
    return grad_obj_function(G, c, input_x) - A.T @ lamb_new - mu * A.T @ c_x
    # return grad_obj_function(G, c, input_x) - A.T @ lamb + mu * (A.T @ A @ input_x - A.T @ b)


def find_alpha(x, lamb, mu, p, alpha_thd=1e-3):
    c1 = 1e-5
    c2 = 0.9
    alpha = 1

    x_new = x + alpha * p

    flag1 = aug_obj(x_new, lamb, mu) >= aug_obj(x, lamb, mu) + c1 * alpha * aug_grad(x, lamb, mu).T @ p
    flag2 = aug_grad(x_new, lamb, mu).T @ p <= c2 * aug_grad(x, lamb, mu, ).T @ p
    while flag1 or flag2:
        alpha *= 0.5
        x_new = x + alpha * p
        if alpha < alpha_thd: return alpha
    return alpha


def find_sub_solution(init_x, lamb, mu, omega):
    x = init_x
    max_iter = 1_000_000
    H = np.eye(len(x))

    for i in range(max_iter):
        g_k = aug_grad(x, lamb, mu)

        if np.linalg.norm(aug_grad(x, lamb, mu), ord=1) <= omega:
            return x, i
        else:
            p = -H @ g_k
            # todo :check find_alpha
            alpha = find_alpha(x, lamb, mu, p)
            s = alpha * p
            x_new = x + alpha * p
            y = aug_grad(x_new, lamb, mu) - aug_grad(x, lamb, mu)

            r = 1 / (y.T @ s)
            li = (np.eye(len(x)) - (r * (s @ y.T)))
            ri = (np.eye(len(x)) - (r * (y @ s.T)))
            hess_inter = li @ H @ ri
            H = hess_inter + (r * (s @ s.T))  # BFGS Update

            x = x_new
            # print(f'{i}:{x.T}  f(x):{obj_function(G, c, x)} mu :{mu} lamb :{lamb.T}')

    print(f"\n!Terminate! Local optimal solution of the augmented function is  not found in {max_iter} iterations")
    return x, max_iter


def aug_lag(input_x):
    x = input_x

    lamb = 1 * np.ones_like(b)
    eta_thd = 1e-5
    omega_thd = 1e-5

    # set init condition
    mu = 1
    omega = mu ** (-1)
    eta = mu ** (-0.1)

    # set iteration config
    max_iter = 1000

    for i in range(max_iter):
        print(f'-----------------{i}-----------------')
        x_sub, iter_num = find_sub_solution(x, lamb, mu, omega)

        c_xk = A @ x_sub - b
        c_ = np.where(c_xk > 0, 0, -c_xk)
        if np.linalg.norm(c_, ord=1) <= eta:
            # Test convergence conditions
            if np.linalg.norm(c_, ord=1) <= eta_thd and \
                    np.linalg.norm(aug_grad(x_sub, lamb, mu, ), ord=1) <= omega_thd:
                print(f'\n======> iter_num {iter_num} \n======> Get correct x*: {x_sub.T}  f(x):{obj_function(G,c,x_sub)}')
                return x_sub
            # Update multiplier to reduce tolerance error
            lamb = lamb - mu * (A @ x_sub - b)
            mu = mu
            eta = eta * mu ** (-0.9)
            omega = omega * mu ** (-1)
        else:
            # Increase penalty function and reduce tolerance error
            lamb = lamb
            mu = 100 * mu
            eta = mu ** (-0.1)
            omega = mu ** (-1)
        print(f'======> iter_num {iter_num} \n======> x: {x_sub.T}  f(x):{obj_function(G, c, x_sub)}')
        x = x_sub
    print(f'\n========> The program did not find the optimal solution after {max_iter} iteration')


if __name__ == '__main__':
    # objection function
    G = np.array([[2, -2],
                  [-2, 4]])
    c = np.array([[-2, -6]]).transpose()

    # constraint
    A = np.array([[-0.5, -1],
                  [1, -2],
                  [1, 0],
                  [0, 1]])
    b = np.array([[-1, -2, 0, 0]]).transpose()
    func_dic = {
        'G': G,
        'c': c,
        'A': A,
        'b': b
    }
    init_x = np.array([[1/4, 1/3]]).transpose()
    solution3 = aug_lag(init_x)
