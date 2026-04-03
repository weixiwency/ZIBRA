#include <mpi.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <limits>
#include <iomanip>

static std::vector<double> LG_TABLE;
static constexpr double E_INV = 0.36787944117144233; 
static constexpr double D_CONST = 1.0 - E_INV; // d = 1 - e^-1
static constexpr double CORR_EPS = 1e-10;

void init_fast_math(int max_y = 20000) {
    if (LG_TABLE.empty()) {
        LG_TABLE.resize(max_y + 1);
        for (int i = 0; i <= max_y; ++i) {
            LG_TABLE[i] = std::lgamma(i + 1.0);
        }
    }
}

inline double fast_lgamma(double y) {
    int iy = static_cast<int>(y);
    return (iy >= 0 && iy <= 20000) ? LG_TABLE[iy] : std::lgamma(y + 1.0);
}

// 泊松分布专用的数学与梯度计算
inline double clamp_mu(double mu) { return std::max(1e-6, std::min(mu, 1e4)); }

inline double get_c_pois(double mu) {
    return std::exp(-D_CONST * clamp_mu(mu));
}

inline double dc_dmu_pois(double mu) {
    return -D_CONST * get_c_pois(mu);
}

inline double dlog_pois_dmu(double y, double mu) {
    return y / mu - 1.0;
}

inline double fast_pois_logpmf(double y, double mu) {
    return y * std::log(mu) - mu - fast_lgamma(y);
}

inline double famoye_term_pois(double exp_neg_y, double mu) {
    return exp_neg_y - get_c_pois(mu);
}

struct FitCache {
    Eigen::VectorXd y1, y2;
    Eigen::ArrayXd exp_neg_y1, exp_neg_y2;
    std::vector<char> mask_y1_zero, mask_y2_zero, mask_both_zero;
    int N;

    FitCache(const Eigen::VectorXd& vy1, const Eigen::VectorXd& vy2) : y1(vy1), y2(vy2), N(vy1.size()) {
        exp_neg_y1.resize(N); exp_neg_y2.resize(N);
        mask_y1_zero.resize(N); mask_y2_zero.resize(N); mask_both_zero.resize(N);
        for (int i = 0; i < N; ++i) {
            exp_neg_y1[i] = std::exp(-y1[i]); exp_neg_y2[i] = std::exp(-y2[i]);
            mask_y1_zero[i] = (y1[i] == 0); mask_y2_zero[i] = (y2[i] == 0);
            mask_both_zero[i] = (y1[i] == 0 && y2[i] == 0);
        }
    }
};

struct NLoptDataPois {
    const FitCache* cache;
    const Eigen::VectorXd* w;
    const Eigen::VectorXd* w_corr;
    const Eigen::ArrayXd* term_fixed;
    double lam;
    bool is_g1;
};

// 目标函数降维：现在 x 向量只有 1 个维度 (mu)
double nlopt_obj_func_pois(const std::vector<double> &x, std::vector<double> &grad, void *f_data) {
    NLoptDataPois* d = reinterpret_cast<NLoptDataPois*>(f_data);
    double mu = clamp_mu(x[0]);
    double c = get_c_pois(mu), dc_mu = dc_dmu_pois(mu);
    
    double total_ll = 0.0, g_mu = 0.0;
    const double penalty_coef = 1e6; double penalty = 0.0, pen_gmu = 0.0;

    for (int i = 0; i < d->cache->N; ++i) {
        double y = d->is_g1 ? d->cache->y1[i] : d->cache->y2[i];
        double exp_neg_y = d->is_g1 ? d->cache->exp_neg_y1[i] : d->cache->exp_neg_y2[i];
        
        double ll_pois = fast_pois_logpmf(y, mu);
        double term1 = exp_neg_y - c;
        double corr_raw = 1.0 + d->lam * term1 * (*d->term_fixed)[i];
        
        double dcorr_dmu = d->lam * (*d->term_fixed)[i] * dc_mu;

        double corr = corr_raw;
        if (corr < CORR_EPS) {
            double diff = CORR_EPS - corr_raw;
            penalty += penalty_coef * diff * diff;
            pen_gmu += -2.0 * penalty_coef * diff * dcorr_dmu;
            corr = CORR_EPS;
        }

        total_ll += (*d->w)[i] * ll_pois + (*d->w_corr)[i] * std::log(corr);

        if (!grad.empty() && corr_raw >= CORR_EPS) {
            g_mu += (*d->w)[i] * dlog_pois_dmu(y, mu) + (*d->w_corr)[i] * (dcorr_dmu / corr);
        }
    }

    if (!grad.empty()) {
        grad[0] = -(g_mu - pen_gmu);
    }
    return -(total_ll - penalty);
}

double loss_lam_scalar(const Eigen::VectorXd& w_corr, const Eigen::ArrayXd& A, double lam, double lambda_reg) {
    double nll = 0.0;
    for (int i = 0; i < A.size(); ++i) {
        double corr = 1.0 + lam * A[i];
        if (corr <= CORR_EPS) return 1e50;
        nll -= w_corr[i] * std::log(corr);
    }
    return nll + lambda_reg * lam * lam;
}

double optimize_lambda(const Eigen::VectorXd& w_corr, const Eigen::ArrayXd& A, double lambda_reg) {
    double lb = -10.0, ub = 10.0;
    for (int i = 0; i < A.size(); ++i) {
        if (A[i] > 0) lb = std::max(lb, (-1.0 + CORR_EPS) / A[i]);
        if (A[i] < 0) ub = std::min(ub, (1.0 - CORR_EPS) / -A[i]);
    }
    if (lb >= ub) return 0.0;

    const double gr = 0.618033988749895;
    double a = lb, b = ub, c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = loss_lam_scalar(w_corr, A, c, lambda_reg), fd = loss_lam_scalar(w_corr, A, d, lambda_reg);

    for (int iter = 0; iter < 50; ++iter) {
        if (std::abs(b - a) < 1e-5) break;
        if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = loss_lam_scalar(w_corr, A, c, lambda_reg); } 
        else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = loss_lam_scalar(w_corr, A, d, lambda_reg); }
    }
    return 0.5 * (a + b);
}

// 基于泊松分布的极简相关系数计算
double compute_rho_pois_cpp(double mu1, double mu2, double lam) {
    double c1 = get_c_pois(mu1);
    double c2 = get_c_pois(mu2);
    return lam * std::sqrt(mu1 * mu2) * c1 * c2 * D_CONST * D_CONST;
}

double lrt_pvalue_df1(double ll_full, double ll_constrained) {
    double D = std::max(2.0 * (ll_full - ll_constrained), 0.0);
    return 0.5 * std::erfc(std::sqrt(D / 2.0));
}

std::string get_relationship(double pval_p1, double pval_p2, double pval_p3, double rho, double alpha = 1e-8) {
    bool accept_p2 = (pval_p2 > alpha);
    bool accept_p3 = (pval_p3 > alpha);

    if (pval_p1 > alpha) {
        return "Mutual Exclusivity";
    } else if (accept_p2 && accept_p3) {
        return "Binary Co-expression";
    } else if (accept_p2) {
        return "A Contains B";
    } else if (accept_p3) {
        return "B Contains A";
    } else if (std::abs(rho) > 0.05) {
        return rho > 0 ? "Continuous Synergistic" : "Continuous Antagonistic";
    } else {
        return "Independent";
    }
}

Eigen::Vector4d apply_constraints(Eigen::Vector4d pis, const std::string& cons) {
    if (cons == "p2_0") pis[1] = 1e-4; else if (cons == "p3_0") pis[2] = 1e-4; else if (cons == "p1_0") pis[0] = 1e-4;
    double s = pis.sum(); return (s <= 0) ? Eigen::Vector4d(0, 0, 0, 1) : pis / s;
}

struct FitResultPois { double mu1, mu2, lam, ll; Eigen::Vector4d pis; };

FitResultPois bzip_fit_mpi(const FitCache& cache, double mu1_init, double mu2_init, 
                           const std::string& constraint, double lambda_reg) {
    Eigen::Vector4d pis(0.25, 0.25, 0.25, 0.25); 
    pis = apply_constraints(pis, constraint);
    double lam = 0.0, curr_ll = -1e18, prev_ll = -1e18;
    double mu1 = mu1_init, mu2 = mu2_init;

    nlopt::opt opt(nlopt::LD_LBFGS, 1); // 降维至 1
    std::vector<double> lb = {1e-4}, ub = {10000.0};
    opt.set_lower_bounds(lb); opt.set_upper_bounds(ub);
    opt.set_xtol_rel(1e-4);

    Eigen::MatrixXd gamma(cache.N, 4);
    Eigen::ArrayXd term1_f(cache.N), term2_f(cache.N);

    for (int iter = 0; iter < 30; ++iter) {
        // E-Step
        curr_ll = 0.0;
        for (int i = 0; i < cache.N; ++i) {
            double p1 = fast_pois_logpmf(cache.y1[i], mu1);
            double p2 = fast_pois_logpmf(cache.y2[i], mu2);
            term1_f[i] = famoye_term_pois(cache.exp_neg_y1[i], mu1);
            term2_f[i] = famoye_term_pois(cache.exp_neg_y2[i], mu2);
            
            double corr = std::max(1.0 + lam * term1_f[i] * term2_f[i], CORR_EPS);
            
            double lp1 = p1 + p2 + std::log(corr) + std::log(std::max(pis[0], 1e-12));
            double lp2 = cache.mask_y2_zero[i] ? (p1 + std::log(std::max(pis[1], 1e-12))) : -1e18;
            double lp3 = cache.mask_y1_zero[i] ? (p2 + std::log(std::max(pis[2], 1e-12))) : -1e18;
            double lp4 = cache.mask_both_zero[i] ? std::log(std::max(pis[3], 1e-12)) : -1e18;

            double max_l = std::max({lp1, lp2, lp3, lp4});
            double sum_e = std::exp(lp1 - max_l) + std::exp(lp2 - max_l) + std::exp(lp3 - max_l) + std::exp(lp4 - max_l);
            curr_ll += max_l + std::log(sum_e);

            gamma(i, 0) = std::exp(lp1 - max_l) / sum_e; gamma(i, 1) = std::exp(lp2 - max_l) / sum_e;
            gamma(i, 2) = std::exp(lp3 - max_l) / sum_e; gamma(i, 3) = std::exp(lp4 - max_l) / sum_e;
        }

        if (iter > 0 && std::abs(curr_ll - prev_ll) < 1e-4) break;
        prev_ll = curr_ll;

        pis = apply_constraints(gamma.colwise().mean(), constraint);
        Eigen::VectorXd w_corr = gamma.col(0), w1 = gamma.col(0) + gamma.col(1), w2 = gamma.col(0) + gamma.col(2);

        std::vector<double> x1 = {mu1}, x2 = {mu2};
        double minf;

        // 优化 mu1
        NLoptDataPois d1 = {&cache, &w1, &w_corr, &term2_f, lam, true};
        opt.set_min_objective(nlopt_obj_func_pois, &d1);
        try { opt.optimize(x1, minf); mu1 = x1[0]; } catch(...) {}
        for(int i=0; i<cache.N; ++i) term1_f[i] = famoye_term_pois(cache.exp_neg_y1[i], mu1); 

        // 优化 mu2
        NLoptDataPois d2 = {&cache, &w2, &w_corr, &term1_f, lam, false};
        opt.set_min_objective(nlopt_obj_func_pois, &d2);
        try { opt.optimize(x2, minf); mu2 = x2[0]; } catch(...) {}
        for(int i=0; i<cache.N; ++i) term2_f[i] = famoye_term_pois(cache.exp_neg_y2[i], mu2); 

        Eigen::ArrayXd A = term1_f * term2_f;
        lam = optimize_lambda(w_corr, A, lambda_reg);
    }
    return {mu1, mu2, lam, curr_ll, pis};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    init_fast_math(); 

    if (rank == 0) std::cout << "ZIBRA-Poisson MPI Engine Initialized. Cores: " << size << std::endl;

    FILE* f_mat = fopen("matrix.bin", "rb");
    if (!f_mat) {
        if (rank == 0) std::cerr << "Error: Cannot open matrix.bin!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int N_cells, N_genes;
    fread(&N_cells, sizeof(int), 1, f_mat);
    fread(&N_genes, sizeof(int), 1, f_mat);
    
    std::vector<double> flat_matrix(N_cells * N_genes);
    fread(flat_matrix.data(), sizeof(double), N_cells * N_genes, f_mat);
    fclose(f_mat);

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;
    Eigen::Map<RowMatrixXd> X(flat_matrix.data(), N_cells, N_genes);

    FILE* f_task = fopen("tasks.bin", "rb");
    if (!f_task) {
        if (rank == 0) std::cerr << "Error: Cannot open tasks.bin!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N_tasks;
    fread(&N_tasks, sizeof(int), 1, f_task);
    
    // 注意：这里的 TaskData 结构体已更改！生成端必须同步改为仅打包 idx_A, idx_B, mu1, mu2。
    struct TaskDataPois {
        int idx_A, idx_B;
        double mu1, mu2;
    };
    std::vector<TaskDataPois> all_tasks(N_tasks);
    fread(all_tasks.data(), sizeof(TaskDataPois), N_tasks, f_task);
    fclose(f_task);

    if (rank == 0) std::cout << "Data loaded successfully. Total Tasks: " << N_tasks << std::endl;

    std::string out_name = "output_chunks/results_rank_" + std::to_string(rank) + ".csv";
    std::ofstream out_file(out_name);
    // CSV Header 同样精简了 m1, t1, m2, t2 
    out_file << "idx_A,idx_B,Relationship,rho,ll_full,pval_p2,pval_p3,pval_p1,mu1,mu2,lam,pi1,pi2,pi3,pi4\n";

    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();

    for (int k = rank; k < N_tasks; k += size) {
        const auto& task = all_tasks[k];
        
        Eigen::VectorXd yA = X.col(task.idx_A);
        Eigen::VectorXd yB = X.col(task.idx_B);
        FitCache cache(yA, yB);
        
        FitResultPois res_f  = bzip_fit_mpi(cache, task.mu1, task.mu2, "None", 0.0);
        FitResultPois res_p1 = bzip_fit_mpi(cache, res_f.mu1, res_f.mu2, "p1_0", 0.0);
        FitResultPois res_p2 = bzip_fit_mpi(cache, res_f.mu1, res_f.mu2, "p2_0", 0.0);
        FitResultPois res_p3 = bzip_fit_mpi(cache, res_f.mu1, res_f.mu2, "p3_0", 0.0);

        double pval_p1 = lrt_pvalue_df1(res_f.ll, res_p1.ll);
        double pval_p2 = lrt_pvalue_df1(res_f.ll, res_p2.ll);
        double pval_p3 = lrt_pvalue_df1(res_f.ll, res_p3.ll);

        double rho_val = compute_rho_pois_cpp(res_f.mu1, res_f.mu2, res_f.lam);
        std::string relation = get_relationship(pval_p1, pval_p2, pval_p3, rho_val, 1e-8);
        
        out_file << task.idx_A << "," << task.idx_B << "," 
                 << relation << ","
                 << std::fixed << std::setprecision(6)
                 << rho_val << "," << res_f.ll << "," 
                 << std::scientific << pval_p2 << "," << pval_p3 << "," << pval_p1 << ","
                 << std::fixed << std::setprecision(6)
                 << res_f.mu1 << "," << res_f.mu2 << "," << res_f.lam << ","
                 << res_f.pis(0) << "," << res_f.pis(1) << "," << res_f.pis(2) << "," << res_f.pis(3) << "\n";
   
        if (rank == 0 && k > 0 && k % (size * 10) == 0) {
            double current_time = MPI_Wtime();
            double elapsed = current_time - start_time;                 
            double progress_ratio = static_cast<double>(k) / N_tasks;   
            double estimated_total = elapsed / progress_ratio;          
            double eta_seconds = estimated_total - elapsed;             

            std::cout << "Progress: " << std::fixed << std::setprecision(2) << (progress_ratio * 100.0) << "%"
                      << " | Time Elapsed: " << (int)elapsed << "s"
                      << " | Estimated Remaining: " << (int)eta_seconds << "s"
                      << " | Estimated Total Time: " << (int)estimated_total << "s" << std::endl;
        }
    }

    out_file.close();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double actual_total_time = MPI_Wtime() - start_time;
        std::cout << "Actual Total Time: " << actual_total_time << " seconds.\n";
    }
    
    MPI_Finalize();
    return 0;
}