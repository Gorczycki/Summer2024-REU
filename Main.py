import numpy as np
import iisignature
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def W_t(num_brownian_motions, time_steps, dt):
    rng = np.random.default_rng(85)
    Z = rng.normal(0, 1, (num_brownian_motions, time_steps))
    W = np.cumsum(np.sqrt(dt) * Z, axis=1)
    return W

def brownianmotion(S_0, stock_mu, W, stock_sigma, T):
    t = np.linspace(0, T, W.shape[1])
    S = S_0 * np.exp((stock_mu - 0.5 * stock_sigma**2) * t + stock_sigma * W)
    return S

def generate_paths(num_paths, time_steps, S_0, stock_mu, stock_sigma, T, dt):
    W = W_t(num_paths, time_steps, dt)
    paths = brownianmotion(S_0, stock_mu, W, stock_sigma, T)
    time = np.linspace(0, T, time_steps)
    time_2d = np.tile(time, (num_paths, 1))  # Repeat time for each path
    paths_2d = np.stack((time_2d, paths), axis=-1)  # Stack time and paths
    return paths_2d

def compute_signatures(paths, degrees):
    signatures = []
    for path in paths:
        sig = iisignature.sig(path, degrees)
        signatures.append(sig)
    return np.array(signatures)

def compute_basis_vector(I, X):
    basis_vector = np.prod([np.power(X, i) for i in I], axis=0)
    return basis_vector

def compute_sig_index(deg, I):
    index = 0
    for i in range(deg):
        index += I[i]
    return index

def generate_multi_indices(degree, dimension):
    indices = []
    
    def recursive_index(level, current, indices):
        if level == 0:
            indices.append(tuple(current))
            return
        for i in range(dimension):
            recursive_index(level - 1, current + [i], indices)
    
    recursive_index(degree, [], indices)
    return indices

def compute_signature_values(signatures, degrees, paths):
    d = paths.shape[2]  # Dimensionality of paths (2D: time and stock price)
    n_paths = paths.shape[0]
    
    signature_values = []
    for i in range(n_paths):
        signature = signatures[i]
        sig_vals = []
        for deg in range(1, degrees + 1):
            indices = generate_multi_indices(deg, d)
            for I in indices:
                sig_idx = sum(iisignature.siglength(d, j) for j in range(1, deg)) + compute_sig_index(deg, I)
                if sig_idx < len(signature):
                    basis_vector = compute_basis_vector(I, paths[i])
                    inner_product = np.inner(basis_vector, paths[i])
                    sig_vals.append(signature[sig_idx] * inner_product)
        signature_values.append(sig_vals)
    
    return np.array(signature_values)


def main():
    # Coefficients and constants
    T = 0.8  # maturity
    S_0 = 49
    stock_mu = 0.25
    stock_sigma = 0.20
    dt = 0.01
    time_steps = int(T / dt)  # number of time steps
    num_brownian_motions = 2  # number of paths (adjust as needed)
    degrees = 2  # degrees of signature

    # Generate paths
    paths = generate_paths(num_brownian_motions, time_steps, S_0, stock_mu, stock_sigma, T, dt)

    # Compute signatures
    #X_hat:
    signatures = compute_signatures(paths, degrees)

    # Compute signature values
    #Model:
    signature_values = compute_signature_values(signatures, degrees, paths)
    

    # Flatten paths and signature values for regression
    X = signature_values.reshape((num_brownian_motions, -1))  # Flatten signature values
    y = np.ones((num_brownian_motions, 1))  # Target values, can be adjusted as needed

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    ell_phi = model.intercept_
    ell_I = model.coef_[0]


    print(signatures)


    """"    

    # Print shapes to debug
    print("Signatures:", signatures)
    print("Signatures shape:", signatures.shape)
    print("Signature values shape:", signature_values.shape)
    print("Paths shape:", paths.shape)

    t = np.linspace(0, T, time_steps)
    plt.figure(figsize=(10, 6))
    for i in range(num_brownian_motions):
        plt.plot(t, paths[i, :, 1], label=f'Path {i+1}')  # Only plot the stock price (second column)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Prices Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    """

    

if __name__ == "__main__":
    main()

