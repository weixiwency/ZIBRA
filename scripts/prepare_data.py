import numpy as np
import scanpy as sc
import itertools
from tqdm import tqdm
import struct
import warnings

warnings.filterwarnings('ignore')

class Smart_Initializer:
    @staticmethod
    def get_marginal_zinb_params(y):
        n = len(y)
        mean_val = np.mean(y)
        var_val = np.var(y)
        p0 = np.sum(y == 0) / n
        pi_hat = min(p0, 0.95)
        mu_nb = mean_val / (1 - pi_hat + 1e-6)
        inflation_term = pi_hat * (1 - pi_hat) * (mu_nb ** 2)
        var_nb = (var_val - inflation_term) / (1 - pi_hat + 1e-6)
        var_nb = max(var_nb , mu_nb + 1e-4)
        m_init = (var_nb - mu_nb) / (mu_nb ** 2 + 1e-6) if var_nb > mu_nb else 1.0
        theta_init = (m_init * mu_nb) / (1 + m_init * mu_nb)
        return max(m_init, 0.1), max(theta_init, 0.05)

def main():
    print("1. Loading data...")
    adata = sc.read_h5ad('/your/path/to/data.h5ad') # Update this path to your actual data file
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = np.array(adata.var_names)

    print("2. Filtering genes...")
    passed_indices = np.where(np.mean(X, axis=0) > 0.01)[0].tolist()

    with open("gene_names.txt", "w") as f:
        for idx in passed_indices:
            f.write(f"{idx},{gene_names[idx]}\n")

    print(f"Passed genes: {len(passed_indices)}")
    if len(passed_indices) < 2: return

    X_passed = np.ascontiguousarray(X[:, passed_indices], dtype=np.float64)
    N_cells, N_genes = X_passed.shape

    print("3. Exporting Expression Matrix to matrix.bin...")
    with open("matrix.bin", "wb") as f:
        f.write(struct.pack('ii', N_cells, N_genes))
        f.write(X_passed.tobytes())

    print("4. Calculating marginals and exporting Tasks to tasks.bin...")
    marginals = [Smart_Initializer.get_marginal_zinb_params(X_passed[:, i]) for i in range(N_genes)]

    gene_pairs = list(itertools.combinations(range(N_genes), 2))
    N_tasks = len(gene_pairs)

    with open("tasks.bin", "wb") as f:
        f.write(struct.pack('i', N_tasks))
        for i, j in tqdm(gene_pairs):
            m1, t1 = marginals[i]
            m2, t2 = marginals[j]
            f.write(struct.pack('i i d d d d', i, j, m1, t1, m2, t2))

    print("Done! matrix.bin and tasks.bin are ready for C++ Engine.")

if __name__ == "__main__":
    main()