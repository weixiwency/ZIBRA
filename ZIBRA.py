import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln
import itertools
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.special import expit, logit, logsumexp
import warnings

warnings.filterwarnings('ignore')

class Smart_Initializer:
    @staticmethod
    def init_pis_by_counts(y1, y2):
        N = len(y1)
        n1 = np.sum((y1 > 0) & (y2 > 0))  
        n2 = np.sum((y1 > 0) & (y2 == 0)) 
        n3 = np.sum((y1 == 0) & (y2 > 0)) 
        n4 = np.sum((y1 == 0) & (y2 == 0))  
        counts = np.array([n1, n2, n3, n4]) + max(1.0, N*0.01)
        return counts / np.sum(counts)

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


class BZINB_Model:
    def __init__(self, tol=1e-4, max_iter=50, lambda_reg=0.2):
        self.tol = tol
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.params = {}
        self.pis = None

    
    def compute_rho(m1, t1, m2, t2, lam):

        e_inv = np.exp(-1.0)
        c1 = ((1.0 - t1) / (1.0 - t1 * e_inv)) ** (1.0 / m1)
        c2 = ((1.0 - t2) / (1.0 - t2 * e_inv)) ** (1.0 / m2)
        
        mu1 = t1 / (m1 * (1.0 - t1))
        mu2 = t2 / (m2 * (1.0 - t2))
        
        A1 = mu1 * (e_inv - 1.0) / (1.0 - t1 * e_inv)
        A2 = mu2 * (e_inv - 1.0) / (1.0 - t2 * e_inv)
        
        sigma1 = np.sqrt(mu1 + m1 * (mu1 ** 2))
        sigma2 = np.sqrt(mu2 + m2 * (mu2 ** 2))
        
        cov = lam * c1 * c2 * A1 * A2
        return float(cov / (sigma1 * sigma2))
    


    def _lam_feasible_bounds(self, A, eps=1e-10, base_bounds=(-10.0, 10.0)):
        A = np.asarray(A, dtype=float)
        lb, ub = -np.inf, np.inf

        pos = A > 0
        neg = A < 0

        if np.any(pos):
            lb = max(lb, np.max((-1.0 + eps) / A[pos]))

        if np.any(neg):
            ub = min(ub, np.min((1.0 - eps) / (-A[neg])))

        lb = max(lb, base_bounds[0])
        ub = min(ub, base_bounds[1])

        if not np.isfinite(lb) or not np.isfinite(ub) or lb >= ub:
            return None, None
        return lb, ub

    def nb_logpmf(self, y, m, theta):
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        m = np.clip(m, 1e-6, 1e4)
        r = 1.0 / m
        coeff = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        return coeff + y * np.log(theta) + r * np.log(1 - theta)

    def get_famoye_term(self, y, m, t):
        e_inv = np.exp(-1)
        base = (1 - t) / (1 - t * e_inv)
        c = base ** (1 / m)
        return np.exp(-y) - c

    def bnb_logpmf(self, y1, y2, m1, t1, m2, t2, lam):
        log_p1 = self.nb_logpmf(y1, m1, t1)
        log_p2 = self.nb_logpmf(y2, m2, t2)
        term1 = self.get_famoye_term(y1, m1, t1)
        term2 = self.get_famoye_term(y2, m2, t2)
        correction = np.maximum(1 + lam * term1 * term2, 1e-10)
        return log_p1 + log_p2 + np.log(correction)

    def _apply_constraint_to_pis(self, pis, constraint):
        pis = np.array(pis, dtype=float)

        if constraint == 'p2_0':
            pis[1] = 1e-4
        elif constraint == 'p3_0':
            pis[2] = 1e-4
        elif constraint == 'p1_0':
            pis[0] = 1e-4

        s = pis.sum()
        if s <= 0:
            pis = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            pis = pis / s
        return pis


    def fit(self, y1, y2, constraint=None, init_params=None):
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)

        self.pis = Smart_Initializer.init_pis_by_counts(y1, y2)
        self.pis = self._apply_constraint_to_pis(self.pis, constraint)

        if init_params is None:
            m1_in, t1_in = Smart_Initializer.get_marginal_zinb_params(y1)
            m2_in, t2_in = Smart_Initializer.get_marginal_zinb_params(y2)
            self.params = {'m1': m1_in, 't1': t1_in, 'm2': m2_in, 't2': t2_in, 'lam': 0.0}
        else:
            self.params = init_params.copy()

        log_lik_history = []

        for iteration in range(self.max_iter):
            lp1 = self.bnb_logpmf(
                y1, y2,
                self.params['m1'], self.params['t1'],
                self.params['m2'], self.params['t2'],
                self.params['lam']
            )
            p1 = np.exp(lp1) * self.pis[0]

            p2 = np.exp(self.nb_logpmf(y1, self.params['m1'], self.params['t1'])) * (y2 == 0) * self.pis[1]
            p3 = np.exp(self.nb_logpmf(y2, self.params['m2'], self.params['t2'])) * (y1 == 0) * self.pis[2]
            p4 = ((y1 == 0) & (y2 == 0)).astype(float) * self.pis[3]

            total_prob = p1 + p2 + p3 + p4 + 1e-20
            gamma = np.vstack([p1, p2, p3, p4]).T / total_prob[:, None]

            self.pis = gamma.mean(axis=0)
            self.pis = self._apply_constraint_to_pis(self.pis, constraint)

            m1, t1 = self.params['m1'], self.params['t1']
            m2, t2 = self.params['m2'], self.params['t2']
            lam = self.params['lam']

            w_corr = gamma[:, 0]
            w1 = gamma[:, 0] + gamma[:, 1]
            w2 = gamma[:, 0] + gamma[:, 2]


            term2_fixed = self.get_famoye_term(y2, m2, t2)

            def loss_g1(p):
                ll_nb = np.sum(w1 * self.nb_logpmf(y1, p[0], p[1]))
                term1 = self.get_famoye_term(y1, p[0], p[1])
                corr = np.maximum(1 + lam * term1 * term2_fixed, 1e-10)
                ll_corr = np.sum(w_corr * np.log(corr))
                return -(ll_nb + ll_corr)

            res1 = minimize(
                loss_g1, [m1, t1],
                bounds=[(0.01, 20), (0.01, 0.999)],
                method='L-BFGS-B'
            )
            self.params['m1'], self.params['t1'] = res1.x

            term1_fixed = self.get_famoye_term(y1, self.params['m1'], self.params['t1'])

            def loss_g2(p):
                ll_nb = np.sum(w2 * self.nb_logpmf(y2, p[0], p[1]))
                term2 = self.get_famoye_term(y2, p[0], p[1])
                corr = np.maximum(1 + lam * term1_fixed * term2, 1e-10)
                ll_corr = np.sum(w_corr * np.log(corr))

                return -(ll_nb + ll_corr)

            res2 = minimize(
                loss_g2, [m2, t2],
                bounds=[(0.01, 20), (0.01, 0.999)],
                method='L-BFGS-B'
            )
            self.params['m2'], self.params['t2'] = res2.x

            term1_final = self.get_famoye_term(y1, self.params['m1'], self.params['t1'])
            term2_final = self.get_famoye_term(y2, self.params['m2'], self.params['t2'])


            A = term1_final * term2_final
            eps_corr = 1e-10
            lb, ub = self._lam_feasible_bounds(A, eps=eps_corr, base_bounds=(-10.0, 10.0))
            if lb is None:
                self.params['lam'] = 0.0
            else:
                lam0 = float(np.clip(self.params['lam'], lb, ub))

                def loss_lam(l):
                    corr = 1.0 + l[0] * A
                    if np.any(corr <= eps_corr):
                        return 1e50
                    nll = -np.sum(gamma[:, 0] * np.log(corr))
                    penalty = self.lambda_reg * (l[0] ** 2)
                    return nll + penalty

                res_lam = minimize(loss_lam, [lam0], bounds=[(lb, ub)], method='L-BFGS-B')
                self.params['lam'] = float(res_lam.x[0])

            curr_ll = np.sum(np.log(total_prob))
            log_lik_history.append(curr_ll)
            if iteration > 0 and abs(log_lik_history[-1] - log_lik_history[-2]) < self.tol:
                break

        return self.params, self.pis, curr_ll



def step1_univariate_filter(X, gene_names, corr_threshold=0.8):
    n_genes = X.shape[1]
    exog = np.ones((X.shape[0], 1))
    passed_genes_indices = []
    
    for i in tqdm(range(n_genes), desc="Univariate Filter"):
        y = X[:, i]
        if np.mean(y) < 0.05: continue 
        try:
            model = ZeroInflatedNegativeBinomialP(y, exog, exog, inflation='logit', p=2)
            res = model.fit(method='nm', maxiter=200, disp=0)
            if not res.mle_retvals['converged']: continue
                
            cov_matrix = res.cov_params()
            std_devs = np.sqrt(np.diag(cov_matrix))
            if np.any(std_devs == 0): continue 
            
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            corr_pi_mu = abs(corr_matrix.iloc[0, 1] if hasattr(corr_matrix, 'iloc') else corr_matrix[0, 1])
            corr_pi_alpha = abs(corr_matrix.iloc[0, 2] if hasattr(corr_matrix, 'iloc') else corr_matrix[0, 2])
            
            if corr_pi_mu < corr_threshold and corr_pi_alpha < corr_threshold:
                passed_genes_indices.append(i)
        except:
            continue
            
    return passed_genes_indices

def lrt_pvalue(ll_full, ll_constrained, df):
    D = max(2 * (ll_full - ll_constrained), 0)
    return 0.5 * stats.chi2.sf(D, df)


def step3_bivariate_decision(yA, yB, marginal_A, marginal_B, alpha=0.05):
    model_full = BZINB_Model()
    model_restr = BZINB_Model()

    init_p = {'m1': marginal_A[0], 't1': marginal_A[1],
              'm2': marginal_B[0], 't2': marginal_B[1], 'lam': 0.0}

    params_f, _, ll_full = model_full.fit(yA, yB, constraint=None, init_params=init_p)
    _, _, ll_p2_0 = model_restr.fit(yA, yB, constraint='p2_0', init_params=params_f)
    _, _, ll_p3_0 = model_restr.fit(yA, yB, constraint='p3_0', init_params=params_f)
    _, _, ll_p1_0 = model_restr.fit(yA, yB, constraint='p1_0', init_params=params_f)

    accept_p2 = lrt_pvalue(ll_full, ll_p2_0, 1) > alpha
    accept_p3 = lrt_pvalue(ll_full, ll_p3_0, 1) > alpha

    if accept_p2 and accept_p3:
        return "Binary Co-expression (共表达)"

    if accept_p2 ^ accept_p3:
        if accept_p2:
            return "A Contains B (A包含B)"
        else:
            return "B Contains A (B包含A)"

    if lrt_pvalue(ll_full, ll_p1_0, 1) > alpha:
        return "Mutual Exclusivity (互斥)"

    rho_val = model_full.compute_rho(
        params_f['m1'], params_f['t1'],
        params_f['m2'], params_f['t2'],
        params_f['lam']
    )

    if abs(rho_val) > 0.05:
        if rho_val > 0:
            return "Continuous Synergistic (连续协同)"
        else:
            return "Continuous Antagonistic (连续拮抗)"

    return "Independent (独立无关)"

def process_single_pair(task_tuple, X, marginals_dict, gene_names):
    i, j, idx_A, idx_B, is_valid = task_tuple
    name_A = gene_names[idx_A]
    name_B = gene_names[idx_B]
    
    if not is_valid:
        return {'Gene_A': name_A, 'Gene_B': name_B, 'Relationship': "Independent (独立无关)"}
        
    yA = X[:, idx_A]
    yB = X[:, idx_B]
    
    relation = step3_bivariate_decision(
        yA, yB, 
        marginal_A=marginals_dict[idx_A], 
        marginal_B=marginals_dict[idx_B], 
        alpha=0.05
    )
    return {'Gene_A': name_A, 'Gene_B': name_B, 'Relationship': relation}


def build_three_matrices(results, all_genes):
    mat_binary = pd.DataFrame(0, index=all_genes, columns=all_genes)
    mat_directed = pd.DataFrame(0, index=all_genes, columns=all_genes)
    mat_continuous = pd.DataFrame(0, index=all_genes, columns=all_genes)

    for row in results:
        gA, gB, rel = row['Gene_A'], row['Gene_B'], row['Relationship']
        
        if gA in mat_binary.index and gB in mat_binary.columns:
            if rel == "Binary Co-expression (共表达)":
                mat_binary.loc[gA, gB] = 1
                mat_binary.loc[gB, gA] = 1
            elif rel == "Mutual Exclusivity (互斥)":
                mat_binary.loc[gA, gB] = -1
                mat_binary.loc[gB, gA] = -1
            elif rel == "A Contains B (A包含B)":
                mat_directed.loc[gA, gB] = 1
            elif rel == "B Contains A (B包含A)":
                mat_directed.loc[gB, gA] = 1
            elif rel == "Continuous Synergistic (连续协同)":
                mat_continuous.loc[gA, gB] = 1
                mat_continuous.loc[gB, gA] = 1
            elif rel == "Continuous Antagonistic (连续拮抗)":
                mat_continuous.loc[gA, gB] = -1
                mat_continuous.loc[gB, gA] = -1

    return mat_binary, mat_directed, mat_continuous


def main():
    try:
        adata = sc.read_h5ad(r'../sim_data.h5ad')
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        gene_names = np.array(adata.var_names)
    except FileNotFoundError:
        return

    passed_indices = step1_univariate_filter(X, gene_names, corr_threshold=0.8)
    passed_genes = gene_names[passed_indices]
    if len(passed_genes) < 2: return
        

    marginals_dict = {}
    for idx in tqdm(passed_indices, desc="Marginals Cache"):
        marginals_dict[idx] = Smart_Initializer.get_marginal_zinb_params(X[:, idx])
        

    X_passed = X[:, passed_indices] 
    corr_matrix, _ = stats.spearmanr(X_passed)
    corr_matrix = np.nan_to_num(corr_matrix) 
    
    X_bin = (X_passed > 0).astype(int)
    intersection = X_bin.T.dot(X_bin) 
    sizes = X_bin.sum(axis=0)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard_matrix = intersection / np.clip(union, 1, None)
    

    gene_pairs = list(itertools.combinations(range(len(passed_indices)), 2))
    tasks = []
    

    valid_count = 0
    for i, j in gene_pairs:
        idx_A = passed_indices[i]
        idx_B = passed_indices[j]
        
        jac_val = jaccard_matrix[i, j]
        corr_val = abs(corr_matrix[i, j])
        
        is_valid = not (jac_val < 0.05 and corr_val < 0.1)
        if is_valid: valid_count += 1
            
        tasks.append((i, j, idx_A, idx_B, is_valid))

    

    n_cores = max(1, mp.cpu_count() - 1)
    
    worker = partial(process_single_pair, X=X, marginals_dict=marginals_dict, gene_names=gene_names)
    
    results = []
    with mp.Pool(processes=n_cores) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks, chunksize=100), total=len(tasks), desc="Parallel Computing"):
            results.append(res)
            

    mat_binary, mat_directed, mat_continuous = build_three_matrices(results, gene_names)
    
    mat_binary.to_csv("Matrix_1_Binary_CoBursting.csv")
    mat_directed.to_csv("Matrix_2_Directional_Inclusion.csv")
    mat_continuous.to_csv("Matrix_3_Continuous_Regulation.csv")

if __name__ == "__main__":
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass
    main()
