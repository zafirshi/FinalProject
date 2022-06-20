import numpy as np


def obj_function(x: np.ndarray) -> float:
    # 转换成二次型
    y = 0.5 * (x.T @ G @ x) + x.T @ c
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

    # G = np.array([[2, 0],
    #               [0, 2]])
    # c = np.array([[-2, -5]]).transpose()
    # A = np.array([[1, -2],
    #               [-1, -2],
    #               [-1, 2],
    #               [1, 0],
    #               [0, 1]])
    # b = np.array([[-2, -6, -2, 0, 0]]).transpose()

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

        # 直接法构造矩阵求p
        zero_array = np.zeros((len(active_set), len(active_set)))

        top = np.concatenate((G, -A_active.T), axis=1)
        bottom = np.concatenate((A_active, zero_array), axis=1)
        M = np.concatenate((top, bottom), axis=0)

        N = np.concatenate((g_k, h_k), axis=0)

        p_lambda = np.linalg.pinv(M) @ N
        p = -p_lambda[:G.shape[1]]
        lamb = p_lambda[-len(active_set):] if len(active_set) > 0 else np.array([])

        thd = 1e-14

        if np.all(np.abs(p) <= thd):
            # 计算拉格朗日乘子
            print('yes')
            lamb_new = np.linalg.pinv(A_active.T) @ g_k  # 不是方正可以使用伪逆
            print(lamb_new)
            if np.any(lamb_new < 0):
                active_set = np.delete(active_set, np.argmin(lamb_new))
                print(active_set)
                print("go home")
            else:
                print('correct x*')
                print(x)
                print(iter_index)
                break
        else:
            # 计算alpha_k,并迭代更新x
            # active set 之外的条件
            # 这里for loop 一个个看
            # 如果这里不是一个个看，会无法验证分母符号，只有分母<0的情况，才将结果和1比谁更小
            mask = A_inactive @ p < 0
            tmp = (b_inactive - A_inactive @ x) / (A_inactive @ p)
            alpha = min(1, np.min(tmp[mask])) if tmp[mask].size > 0 else 1
            if alpha < 1:
                # if 存在 blocking constrains;
                # Obtain W_k+1 by adding one of the blocking constrains to W_k
                # 这里会有bug
                add_index = np.where(tmp == alpha)[0]
                # add_index = np.argwhere(tmp[mask] < 1)  # 返回多个小于1的index，由于只要one of，所以就i把最小的那个index放进去
                # add_index = add_index.squeeze(-1)
                # add_index = np.argwhere(np.min(tmp[mask]))
                active_set = np.append(active_set, inactive_index[add_index])
            x = x + alpha * p

        iter_index += 1
