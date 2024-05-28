import numpy as np
from scipy.stats import beta
from tqdm import tqdm

def generate_x_v1(X_hist, A_t, params):
    X_new = np.dot(params['theta_x'] * A_t, np.mean(X_hist, axis=0)) 
    X_new += np.random.normal(scale=params['noise_scale'], size=X_new.shape)
    return X_new

def generate_x_v2(X_hist, A_t, params):
    non_linear_transforms = [
        lambda x: np.cos(x * 3.14159 * 2.),
        lambda x: x**0.5,
        lambda x: x,
        lambda x: 1 - np.cos(x * 3.14159 * 2.), 
        lambda x: x**2,
        lambda x: np.sin(x * 3.14159 * 2.),  
    ]

    non_linear_A_t = np.array([func(A_t) for func in non_linear_transforms]) 
    # non_linear_A_t *= ( 1 / np.mean(non_linear_A_t))

    X_new = np.dot(X_hist[-1], params['theta_x'] * non_linear_A_t)
    X_new += np.random.normal(scale=params['noise_scale_x'], size=X_new.shape)
    
    return X_new

def generate_x_v3(X_hist, A_t, params):
    non_linear_transforms = [
        lambda x: x,
        lambda x: x**0.5,
        lambda x: x**2,
        lambda x: np.cos(x * 3.14159 * 2.),
        lambda x: 1-np.cos(x * 3.14159 * 2.),
        lambda x: np.sin(x * 3.14159 * 2.)
    ]

    a = np.array([func(A_t) for func in non_linear_transforms]).reshape(-1,)

    x_t = np.mean(X_hist, axis=0)

    masks = [
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1]
    ]

    A_matrix = np.array([a * mask for mask in masks])

    X_new = A_matrix @ x_t
    X_new += np.random.normal(scale=params['noise_scale_x'], size=X_new.shape)
    return X_new

def generate_x(X_hist, A_t, params, mode='v2'):
    if mode == 'v1':
        return generate_x_v1(X_hist, A_t, params)
    elif mode == 'v2':
        return generate_x_v2(X_hist, A_t, params)
    elif mode == 'v3':
        return generate_x_v3(X_hist, A_t, params)
    else:
        raise ValueError('mode must be v1 or v2 or v3')

def generate_y_v1(X_hist, A_t, params):
    Y_new = np.dot(params['theta_y'] * (A_t), np.mean(X_hist, axis=0)) 
    Y_new += np.random.normal(scale=params['noise_scale_y'])
    return Y_new

def generate_y(X_hist, A_t, params):
    x = np.mean(X_hist, axis=0)
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    t = A_t
    Y_new = np.cos(t * 3.14159 *2) * x1 + t**2 * x4 + np.sin(t * 3.14159 * 2) * x6 + np.exp(t) * x3
    Y_new += np.random.normal(scale=params['noise_scale_y'])
    return Y_new
    

def generate_a_v1(X_hist, params):
    return np.random.uniform(0, 1)

def generate_a_v2(X_hist, params):
    x = np.mean(X_hist, axis=0)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    d_t = np.sin(x1 * 3.14159 * 2) + np.cos(x2 * 3.14159 * 2) * x5 + np.max([x3, x4])
    d_t = d_t + np.random.normal(scale=params['noise_scale_a'])
    gamma = params['gamma']
    # sigmoid
    d_t = 1 / (1 + np.exp(-d_t))
    gamma_1 = 1 + gamma * d_t
    gamma_2 = 1 + gamma * (1 - d_t)
    t = beta.rvs(gamma_1, gamma_2)
    return t

def generate_a(X_hist, params, mode='v2'):
    if mode == 'v1':
        return generate_a_v1(X_hist, params)
    elif mode == 'v2':
        return generate_a_v2(X_hist, params)
    else:
        raise ValueError('mode must be v1 or v2')

def simulate_time_series(T, h, params, mode_x='v2', mode_a='v1'):
    X_dim = len(params['theta_x'])
    X = np.random.normal(scale=1, size=(T, X_dim))
    # X = np.random.uniform(0, 1, (T, X_dim))
    Y = np.zeros((T, 1))
    A = np.zeros((T, 1))
    
    for t in range(T - 1):
        start = max(0, t - h + 1)
        A[t] = generate_a(X[start:t + 1], params, mode=mode_a)
        X[t + 1] = generate_x(X[start:t + 1], A[t], params, mode=mode_x)
        Y[t + 1] = generate_y(X[start:t + 1], A[t], params)
    
    return X, Y, A

def simulate_factual(n, seq_len, lag, params, mode_x='v2', mode_a='v1'):
    static_features = np.zeros((n, seq_len, 1))
    X = np.zeros((n, seq_len, 6))
    Y = np.zeros((n, seq_len, 1))
    A = np.zeros((n, seq_len, 1))
    sequence_lengths = np.zeros((n, 1))
    
    for i in range(n):
        X[i], Y[i], A[i] = simulate_time_series(seq_len, lag, params, mode_x=mode_x, mode_a=mode_a)
        sequence_lengths[i] = seq_len - 1

    previous_treatments = np.concatenate((np.zeros((n, 1, 1)), A[:, :-2, :]), axis=1)
    current_treatments = A[:, :-1, :]
    
    # add active entries
    active_entries = np.ones((n, seq_len - 1, 1))
    
    return {'current_covariates': X[:, :-1, :], 
            'next_covariates': X[:, 1:, :],
            'prev_treatments': previous_treatments,
            'current_treatments': current_treatments,
            'active_entries': active_entries,
            'sequence_lengths': sequence_lengths,
            'static_features': static_features[:, 0, :],
            'outputs': Y[:, 1:, :]
        }

def simulate_counterfactual_one_step(n, seq_len, lag, params, counterfactual_points=5, mode_x='v2', mode_a='v1'):
    num_test_points = n * seq_len * counterfactual_points
    static_features = np.zeros((num_test_points, 1))
    X = np.zeros((num_test_points, seq_len - 1, 6))
    Y = np.zeros((num_test_points, seq_len, 1))
    A = np.zeros((num_test_points, seq_len - 1, 1))
    pre_A = np.zeros((num_test_points, seq_len - 1, 1))
    next_X = np.zeros((num_test_points, seq_len - 1, 6))
    sequence_lengths = np.zeros(num_test_points)
    active_entries = np.zeros((num_test_points, seq_len - 1, 1))

    test_idx = 0

    for i in tqdm(range(n), total=n):
        factual_X = np.random.normal(scale=1, size=(seq_len, 6))
        factual_Y = np.zeros((seq_len, 1))
        factual_A = np.zeros((seq_len, 1))
        for t in range(seq_len - 1):
            start = max(0, t - lag + 1)
            factual_A[t] = generate_a(factual_X[start:t + 1], params, mode=mode_a)
            factual_X[t + 1] = generate_x(factual_X[start:t + 1], factual_A[t], params, mode=mode_x)
            factual_Y[t + 1] = generate_y(factual_X[start:t + 1], factual_A[t], params)
            for j in range(counterfactual_points):
                # generate counterfactual A randomly
                counterfactual_A = np.random.uniform(0, 1)
                # generate counterfactual X and Y
                counterfactual_X = generate_x(factual_X[start:t + 1], counterfactual_A, params, mode=mode_x)
                counterfactual_Y = generate_y(factual_X[start:t + 1], counterfactual_A, params)
                X[test_idx, : t + 1, :] = factual_X[: t + 1, :]
                Y[test_idx, : t + 2, :] = np.concatenate((factual_Y[: t + 1, :], counterfactual_Y.reshape(1, 1)), axis=0)
                A[test_idx, : t + 1, :] = np.concatenate((factual_A[: t, :], counterfactual_A * np.ones((1, 1))), axis=0)
                pre_A[test_idx, : t + 1, :] = np.concatenate((np.zeros((1, 1)), factual_A[: t, :]), axis=0)
                next_X[test_idx, : t + 1, :] = np.concatenate((factual_X[1 : t + 1, :], counterfactual_X.reshape(1, 6)), axis=0)
                sequence_lengths[test_idx] = t + 1
                active_entries[test_idx, : t + 1, :] = 1
                test_idx += 1

    return {'current_covariates': X[:test_idx], 
            'next_covariates': next_X[:test_idx],
            'prev_treatments': pre_A[:test_idx],
            'current_treatments': A[:test_idx],
            'active_entries': active_entries[:test_idx],
            'sequence_lengths': sequence_lengths[:test_idx],
            'static_features': static_features,
            'outputs': Y[:test_idx, 1:, :]
        }

def simulate_counterfactuals_treatment_seq(n, seq_len, lag, params, projection_horizon=5, mode_x='v2', mode_a='v1'):
    # generate random intervation sequences with length projection_horizon
    treatment_options = np.random.uniform(0, 1, (projection_horizon, projection_horizon, 1))
    # seq_len = seq_len - projection_horizon
    num_test_points = n * seq_len * treatment_options.shape[0]

    static_features = np.zeros((num_test_points, 1))
    X = np.zeros((num_test_points, seq_len + projection_horizon, 6))
    Y = np.zeros((num_test_points, seq_len + projection_horizon, 1))
    A = np.zeros((num_test_points, seq_len - 1 + projection_horizon, 1))
    pre_A = np.zeros((num_test_points, seq_len - 1 + projection_horizon, 1))
    next_X = np.zeros((num_test_points, seq_len - 1 + projection_horizon, 6))
    sequence_lengths = np.zeros(num_test_points)
    active_entries = np.zeros((num_test_points, seq_len - 1 + projection_horizon, 1))

    test_idx = 0

    for i in tqdm(range(n), total=n):
        factual_X = np.random.normal(scale=1, size=(seq_len, 6))
        factual_Y = np.zeros((seq_len, 1))
        factual_A = np.zeros((seq_len, 1))
        for t in range(seq_len - 1):
            start = max(0, t - lag + 1)
            factual_A[t] = generate_a(factual_X[start:t + 1], params, mode=mode_a)
            factual_X[t + 1] = generate_x(factual_X[start:t + 1], factual_A[t], params, mode=mode_x)
            factual_Y[t + 1] = generate_y(factual_X[start:t + 1], factual_A[t], params)
            for treatment_option in treatment_options:
                # initilize
                # counterfactual_A = np.zeros((projection_horizon, 1))
                # counterfactual_X = np.zeros((projection_horizon, 6))
                # counterfactual_Y = np.zeros((projection_horizon, 1))
                X[test_idx, : t + 2, :] = factual_X[: t + 2, :]
                Y[test_idx, : t + 2, :] = factual_Y[: t + 2, :]
                A[test_idx, : t + 1, :] = factual_A[: t + 1, :]
                for projection_time in range(1, projection_horizon + 1):
                    # generate counterfactual X and Y 
                    start = max(0, t - lag + 1 + projection_time)
                    # counterfactual_A = treatment_option[projection_time - 1]
                    # counterfactual_A = generate_a(X[test_idx, start:t + 1 + projection_time, :], params, mode=mode_a)
                    counterfactual_A = np.random.uniform(0, 1)
                    counterfactual_X = generate_x(X[test_idx, start:t + 1 + projection_time, :], counterfactual_A, params, mode=mode_x)
                    counterfactual_Y = generate_y(X[test_idx, start:t + 1 + projection_time, :], counterfactual_A, params)
                    # check if the counterfactual X and Y are valid
                    if np.isnan(counterfactual_X).any() or np.isnan(counterfactual_Y).any():
                        print('invalid counterfactual X or Y')
                        print(f'counterfactual_A: {counterfactual_A}, counterfactual_X: {counterfactual_X}, counterfactual_Y: {counterfactual_Y}')
                    X[test_idx, t + 1 + projection_time, :] = counterfactual_X
                    Y[test_idx, t + 1 + projection_time, :] = counterfactual_Y
                    A[test_idx, t + projection_time, :] = counterfactual_A
                sequence_lengths[test_idx] = t + 1 + projection_horizon
                active_entries[test_idx, : t + 1 + projection_horizon, :] = 1
                next_X[test_idx, : t + 1 + projection_horizon, :] = X[test_idx, 1 : t + 2 + projection_horizon, :]
                X[test_idx, t + 1 + projection_horizon : , :] = 0
                # print(f'A[test_idx, : t + projection_time, :].shape: {A[test_idx, : t + projection_time, :].shape}')
                pre_A[test_idx, : t + 1 + projection_horizon, :] = np.concatenate((np.zeros((1, 1)), A[test_idx, : t + projection_horizon, :]), axis=0)
                test_idx += 1
                
    return {'current_covariates': X[:test_idx, :-1, :], 
            'next_covariates': next_X[:test_idx],
            'prev_treatments': pre_A[:test_idx],
            'current_treatments': A[:test_idx],
            'active_entries': active_entries[:test_idx],
            'sequence_lengths': sequence_lengths[:test_idx],
            'static_features': static_features[:test_idx],
            'outputs': Y[:test_idx, 1:, :]
        }