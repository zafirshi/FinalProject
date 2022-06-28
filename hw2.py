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


def aug_obj_function(x: np.ndarray, mu, **kwargs):
    tmp = A @ x - b

    out = np.where(tmp > 0, 0, -tmp)
    out = out.T

    back = np.square(np.linalg.norm(out, ord=2))

    return obj_function(G, c, x) + 0.5 * mu * back


def aug_grad_obj_function(x, mu, **kwargs):
    tmp = A @ x - b

    c_ = np.where(tmp > 0, 0, -tmp)

    return grad_obj_function(G, c, x) - 0.5 * mu * 2 * A.T @ c_


def find_alpha(x, mu, alpha_thd=1e-5):
    c1 = 1e-5
    c2 = 0.9
    alpha = 1

    d = len(x)
    H = np.eye(d)

    p_k = -H @ aug_grad_obj_function(x, mu)

    x_new = x + alpha * p_k
    flag1 = aug_obj_function(x_new, mu) >= aug_obj_function(x, mu) + c1 * alpha * aug_grad_obj_function(x, mu).T @ p_k
    flag2 = aug_grad_obj_function(x_new, mu).T @ p_k <= c2 * aug_grad_obj_function(x, mu).T @ p_k
    while flag1 or flag2:
        alpha *= 0.5
        x_new = x + alpha * p_k
        if alpha < alpha_thd: return alpha
    return alpha


def GD(func, init, thd, mu, **kwargs):
    x = init
    max_iter = 1_000_000

    for i in range(max_iter):
        # g_k = grad(func(x))
        # Todo: 完成增广目标函数求梯度
        g_k = aug_grad_obj_function(x, mu)
        # print(g_k)
        if np.linalg.norm(g_k, ord=1) < thd:
            return x, i
        else:
            d_k = -g_k
            alpha = find_alpha(x, mu, )
            # alpha_bk = alpha
        x = x + alpha * d_k
        # print(f'{i}:{x.T}  f(x):{obj_function(G,c,x)}')

    print("算法终止，增广函数的局部最优解在迭代次数中未找到")
    return x, max_iter


def BFGS():
    pass


def newton():
    pass


def get_approximate_solution(aug, mu, tau, init_x, method: str = 'newton'):
    if method.lower() == 'newton':
        newton()
    elif method.upper() == 'BFGS':
        BFGS()
    elif method.upper() == 'GD':
        return GD(aug, init_x, tau, mu)
    else:
        raise NotImplementedError("method is not Implemented")


def quadratic_penalty_method(**kwargs):
    mu = 1  # 惩罚因子
    tau = 1e-5  # 扩充目标函数阈值
    thd = 0.001  # 收敛性检测阈值

    init_x = np.array([[0.2, 0.2]]).transpose()
    assert init_x.shape[1] == 1  # check x whether be a column vector

    max_iter = 100

    object_gradient = grad_obj_function(G, c, init_x)
    aug = aug_obj_function(init_x, mu, **kwargs)

    for epoch in range(max_iter):
        # find approximate solution of penalty function

        print(f'_________________________________________{epoch}_________________________________________')
        x, iter_num = get_approximate_solution(aug_obj_function, mu, tau, init_x, method='GD')
        print(f'iter num: {iter_num}, x: {x.T}, f(x): {obj_function(G, c, x)}')
        # 收敛性测试 -> 简单认为是目标函数数值没有很大变化
        object_function_last = obj_function(G, c, init_x)
        object_function_this = obj_function(G, c, x)

        aug_last = aug_obj_function(init_x, mu, **kwargs)
        aug_this = aug_obj_function(x, mu, **kwargs)

        if np.abs(object_function_last - object_function_this) < thd or iter_num == 10000:
            print(f'x*:{x.T}----f(x):{object_function_this}')
            return x
        # if np.linalg.norm(grad_obj_function(G, c, x), ord=1) < thd:
        #     object_function = obj_function(G, c, x)
        #     print(f'x*:{x}----f(x):{object_function}')
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
    x = quadratic_penalty_method(**func_dic)
