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


def find_alpha(x, mu, p, penalty_type, alpha_thd=1e-2):
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


def GD(func, init, thd, mu, penalty_type, **kwargs):
    x = init
    max_iter = 1_000_000

    H = np.eye(len(x))

    for i in range(max_iter):
        # Todo: 完成增广目标函数求梯度
        g_k = aug_grad_obj_function(x, mu, penalty_type)
        if np.linalg.norm(g_k, ord=1) < thd:
            return x, i
        else:
            p_k = -H @ aug_grad_obj_function(x, mu, penalty_type)

            alpha = find_alpha(x, mu, p_k, penalty_type)

            x_new = x + alpha * p_k
            x = x_new
            print(f'{i}:{x.T}  f(x):{obj_function(G, c, x)}')

    print("算法终止，增广函数的局部最优解在迭代次数中未找到")
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
            print(f'{i}:{x.T}  f(x):{obj_function(G, c, x)} mu :{mu}')

    print("算法终止，增广函数的局部最优解在迭代次数中未找到")
    return x, max_iter


def newton():
    pass


def get_approximate_solution(aug, mu, tau, init_x, penalty_type, method: str = 'newton'):
    if method.lower() == 'newton':
        newton()
    elif method.upper() == 'BFGS':
        return BFGS(aug, init_x, tau, mu, penalty_type)
    elif method.upper() == 'GD':
        return GD(aug, init_x, tau, mu, penalty_type)
    else:
        raise NotImplementedError("method is not Implemented")


def penalty_method(penalty_type, **kwargs):
    mu = 1  # 惩罚因子
    tau = 1e-5  # 扩充目标函数阈值
    thd = 0.001  # 收敛性检测阈值

    init_x = np.array([[0., 0.]]).transpose()
    assert init_x.shape[1] == 1  # check x whether be a column vector

    max_iter = 100

    object_gradient = grad_obj_function(G, c, init_x)
    aug = aug_obj_function(init_x, mu, penalty_type, **kwargs)

    for epoch in range(max_iter):
        # find approximate solution of penalty function

        print(f'_________________________________________{epoch}_________________________________________')
        x, iter_num = get_approximate_solution(aug_obj_function, mu, tau, init_x, penalty_type, method='BFGS')
        print(f'iter num: {iter_num}, x: {x.T}, f(x): {obj_function(G, c, x)}')
        # 收敛性测试 -> 简单认为是目标函数数值没有很大变化
        object_function_last = obj_function(G, c, init_x)
        object_function_this = obj_function(G, c, x)

        aug_last = aug_obj_function(init_x, mu, penalty_type, **kwargs)
        aug_this = aug_obj_function(x, mu, penalty_type, **kwargs)

        if np.abs(object_function_last - object_function_this) < thd:
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
    # solution1 = penalty_method('quadratic', **func_dic)
    solution2 = penalty_method('classic', **func_dic)
    # solution3 = penalty_method('lagrangian', **func_dic)
