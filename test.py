import time
import numpy as np

# used for recording program run time
start_time = time.time()

# ---------- set problem
# transform into quadratic form
G = np.array([[2, 0],
              [0, 2]])
c = np.array([-2, -5])
A0 = np.array([[1, -2],
               [-1, -2],
               [-1, 2],
               [1, 0],
               [0, 1]])
b0 = np.array([-2, -6, -2, 0, 0])


# objective function
def objFun(x):
    y = 1 / 2 * x.T @ G @ x + c @ x
    return y


# ---------- initialization
# initial feasible point
x = np.array([2, 0])
x_num = len(x)


# initial active/inactive set
total = np.arange(len(A0))
active = np.where(np.matmul(A0, x) == b0)[0]
inactive = np.where(np.matmul(A0, x) != b0)[0]

# ---------- solve EQP and Lagrangian multipliers lambda defined by current active set
print('Start Active Set Algorithm')
print('Initial solution:', x)
print('Initial objective function value: ', objFun(x))
itr = 1
while True:
    print('########## Iteration: ', itr, '##########')

    # number of equality
    equ_num = len(active)

    A = A0[active, :]
    b = b0[active]

    # prepare matrix
    zeros = np.zeros((equ_num, equ_num))
    upper = np.concatenate((G, A.T), axis=1)
    lower = np.concatenate((A, zeros), axis=1)
    M = np.concatenate((upper, lower), axis=0)
    v = np.concatenate((-c, b))

    # solve sub problem/EQP based on current active set
    inv = np.linalg.inv(M)
    sol = np.matmul(inv, v)
    # x_eqp solution
    x_eqp = sol[:x_num]
    # lagrangian multipliers
    lamb = sol[-equ_num:]

    # check x_eqp is feasible or not
    # situation 1: x_eqp is feasible
    if (np.matmul(A0, x_eqp) <= b0).all():
        # check whether Lagrangian multipliers are all larger or eqaul to zero
        if (lamb >= 0).all():
            # update x
            x = x_eqp
            print('Current solution: ', x)
            print('Current objective value: ', objFun(x))
            print('=> Optimal solution found')
            break
        else:
            # remove a constraint with minimun negative Lagrangian multiplier from current active set
            index = np.argmin(lamb)
            active = np.delete(active, index)
            inactive = np.setdiff1d(total, active)
            # update x
            x = x_eqp

    # situation 2: x_eqp is not feasible
    else:
        p = x_eqp - x
        candidate = [1]
        # find the maximum stepsize alpha can be
        for i in inactive:
            a_i = A0[i,]
            if np.matmul(a_i, p) > 0:
                b_i = b0[i]
                value = (b_i - np.matmul(a_i, x)) / np.matmul(a_i, p)
                candidate.append(value)
        alpha = min(candidate)

        # update x
        x = x + alpha * p
        active = np.where(np.matmul(A0, x) == b0)[0]
        inactive = np.where(np.matmul(A0, x) != b0)[0]

    print('Current solution: ', x)
    print('Current objective value: ', objFun(x))
    itr += 1

# ---------- algorithm result
print('------------------------------')
print('* Total iterations: ', itr)
print('* Total run time:', round((time.time() - start_time), 4), 'sec')
print('* Optimal solution: ', x)
print('* Optimal objective function value:', objFun(x))
