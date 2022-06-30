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


def aug_obj_function(x: np.ndarray, mu, penalty_type, **kwargs):
    tmp = A @ x - b

    out = np.where(tmp > 0, 0, -tmp)
    out = out.T

    if penalty_type.lower() == 'quadratic':
        back = np.square(np.linalg.norm(out, ord=2))
        return obj_function(G, c, x) + 0.5 * mu * back
    elif penalty_type.lower() == 'classic':
        back = np.sum(out, axis=1).item()
        return obj_function(G, c, x) + mu * back
    else:
        raise NotImplementedError('aug_obj_function: penalty_type is not implemented')


def aug_grad_obj_function(x, mu, penalty_type, **kwargs):
    if penalty_type.lower() == 'quadratic':
        tmp = A @ x - b
        c_ = np.where(tmp > 0, 0, -tmp)
        return grad_obj_function(G, c, x) - 0.5 * mu * 2 * A.T @ c_

    elif penalty_type.lower() == 'classic':
        tmp = A @ x - b
        c_ = np.where(tmp > 0, 0, -tmp)

        c_ = np.where(c_ > 0, 1, c_)
        return grad_obj_function(G, c, x) - A.T @ c_

    else:
        raise NotImplementedError('aug_grad_obj_function: penalty_type is not implemented')


def find_alpha(x, mu, p, penalty_type, alpha_thd=1e-5):
    c1 = 1e-5
    c2 = 0.9
    alpha = 1

    x_new = x + alpha * p

    flag1 = aug_obj_function(x_new, mu, penalty_type) >= \
            aug_obj_function(x, mu, penalty_type) + c1 * alpha * aug_grad_obj_function(x, mu, penalty_type).T @ p
    flag2 = aug_grad_obj_function(x_new, mu, penalty_type).T @ p <= \
            c2 * aug_grad_obj_function(x, mu, penalty_type).T @ p
    while flag1 or flag2:
        alpha *= 0.5
        x_new = x + alpha * p
        if alpha < alpha_thd: return alpha
    return alpha


def GD(aug, init, thd, mu, penalty_type, **kwargs):
    x = init
    max_iter = 1_000_000

    H = np.eye(len(x))

    for i in range(max_iter):
        g_k = aug_grad_obj_function(x, mu, penalty_type)
        if np.linalg.norm(g_k, ord=1) < thd:
            return x, i
        else:
            p_k = -H @ aug_grad_obj_function(x, mu, penalty_type)

            alpha = find_alpha(x, mu, p_k, penalty_type)

            x_new = x + alpha * p_k
            x = x_new
            # print(f'{i}:{x.T}  f(x):{obj_function(G, c, x)}')

    print(f"Terminate without finding aug_function solution in {max_iter} iterations")
    return x, max_iter


def BFGS(aug, init_x, thd, mu, penalty_type, **kwargs):
    x = init_x
    max_iter = 1_000_000
    H = np.eye(len(x))

    for i in range(max_iter):
        g_k = aug_grad_obj_function(x, mu, penalty_type)

        if np.linalg.norm(g_k, ord=1) < thd:
            return x, i
        else:
            p = -H @ g_k
            # todo :check find_alpha
            alpha = find_alpha(x, mu, p, penalty_type)
            # alpha = 0.001
            s = alpha * p
            x_new = x + alpha * p
            y = aug_grad_obj_function(x_new, mu, penalty_type) - aug_grad_obj_function(x, mu, penalty_type)

            r = 1 / (y.T @ s)
            li = (np.eye(len(x)) - (r * (s @ y.T)))
            ri = (np.eye(len(x)) - (r * (y @ s.T)))
            hess_inter = li @ H @ ri
            H = hess_inter + (r * (s @ s.T))  # BFGS Update

            x = x_new
            # print(f'{i}:{x.T}  f(x):{obj_function(G, c, x)} mu :{mu}')

    print(f"=====> Failed! Terminate without finding solution in {max_iter} iterations")
    return x, max_iter


def newton():
    pass


def get_approximate_solution(aug, mu, tau, init_x, penalty_type, method: str = 'newton'):
    if method.lower() == 'newton':
        return newton()
    elif method.upper() == 'BFGS':
        return BFGS(aug, init_x, tau, mu, penalty_type)
    elif method.upper() == 'GD':
        return GD(aug, init_x, tau, mu, penalty_type)
    else:
        raise NotImplementedError("method is not Implemented")


def penalty_method(input_x, penalty_type, method, **kwargs):
    mu = 1  # penalty factor
    tau = 1e-5  # Extended objective function threshold
    thd = 0.001  # Convergence detection threshold

    init_x = input_x
    assert init_x.shape[1] == 1  # check x whether be a column vector

    max_iter = 100

    object_gradient = grad_obj_function(G, c, init_x)
    aug = aug_obj_function(init_x, mu, penalty_type, **kwargs)

    for epoch in range(max_iter):
        # find approximate solution of penalty function

        print(f'_________________________________________{epoch}_________________________________________')
        x, iter_num = get_approximate_solution(aug_obj_function, mu, tau, init_x, penalty_type, method=method)
        print(f'iter num: {iter_num}, x: {x.T}, f(x): {obj_function(G, c, x)}')
        # Convergence test - > simply consider that the value of the objective function has not changed greatly
        object_function_last = obj_function(G, c, init_x)
        object_function_this = obj_function(G, c, x)

        if penalty_type.lower() == 'quadratic':
            if np.abs(object_function_last - object_function_this) < thd:
                print('\n===========================Find correct answer!===========================')
                print(f'======> epoch_num {epoch} correct x*: {x.T}  f(x):{object_function_this}')
                return x

        # If the constraint condition is less than the threshold after the []- operation, the loop is terminated
        elif penalty_type.lower() == 'classic':
            tmp = A @ x - b
            c_x = np.where(tmp > 0, 0, -tmp)

            if np.sum(c_x.squeeze(-1)) < thd:
                print(f'x*:{x.T}----f(x):{object_function_this}')
                return x
        mu = mu * 5
        init_x = x

    print(f'Search terminated cause iteration out of max range{max_iter}')


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
    input_x = np.array([[1., 1 / 2]]).transpose()
    # todo: For the second question, release solution2
    solution1 = penalty_method(input_x, penalty_type='quadratic', method='GD', **func_dic)
    # solution2 = penalty_method('classic', **func_dic)
