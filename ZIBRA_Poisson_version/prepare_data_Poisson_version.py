import numpy as np
import scanpy as sc
import itertools
from tqdm import tqdm
import struct
import warnings

warnings.filterwarnings('ignore')

class Smart_Initializer:
    @staticmethod
    def get_marginal_zip_params(y):
        # 针对零膨胀泊松分布(ZIP)的极简初始化
        n = len(y)
        mean_val = np.mean(y)
        p0 = np.sum(y == 0) / n
        pi_hat = min(p0, 0.95) # 预估的零膨胀比例
        
        # 由于 mean = (1 - pi) * mu，所以推导泊松均值 mu
        mu_init = mean_val / (1 - pi_hat + 1e-6)
        return max(mu_init, 1e-4) # 只返回一个 mu 参数

def main():
    print("1. Loading data...")
    adata = sc.read_h5ad(r'/home/weixi/ZIBRA_Poisson_version/sim_adata.h5ad') # 记得替换路径
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
    # 这里变成了每个基因只返回一个 mu
    marginals = [Smart_Initializer.get_marginal_zip_params(X_passed[:, i]) for i in range(N_genes)]

    gene_pairs = list(itertools.combinations(range(N_genes), 2))
    N_tasks = len(gene_pairs)

    with open("tasks.bin", "wb") as f:
        f.write(struct.pack('i', N_tasks))
        for i, j in tqdm(gene_pairs):
            mu1 = marginals[i]
            mu2 = marginals[j]
            # 格式从 'i i d d d d' 变成了 'i i d d' (两个 int, 两个 double)
            f.write(struct.pack('i i d d', i, j, mu1, mu2))

    print("Done! matrix.bin and tasks.bin are ready for C++ Poisson Engine.")

if __name__ == "__main__":
    main()