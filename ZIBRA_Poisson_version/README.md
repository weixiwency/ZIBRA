# ZIBRA

ZIBRA is a high-performance computing framework for analyzing gene-gene associations in large-scale single-cell RNA sequencing (scRNA-seq) data using the Bivariate Zero-Inflated Poisson (BZINB) model.

## Prerequisites
### 1. Python Environment
```bash
conda create -n bzinb python=3.9
conda activate bzinb
pip install numpy pandas scanpy scipy tqdm statsmodels
```
### 2. C++ & HPC Libraries
* OpenMPI or MPICH
* NLopt
* Eigen 3

## Quick Start
### Step 1: Compile the C++ MPI Engine
Ensure paths to `eigen3` and `nlopt` match your system configuration in the `Makefile`, then compile:

```bash
make
```
### Step 2: Data Preprocessing (Python)
Pack your `.h5ad` single-cell data into binary chunks:

```bash
python scripts/prepare_data.py
```

### Step 3: MPI Computation (C++)
Launch the MPI engine using all available cores (e.g., 124 cores):

```bash
mkdir -p output_chunks
mpirun -n 124 ./zibra_pois_mpi_engine
```

### Step 4: Merge Results (Python)
Merge the distributed output chunks and map the numerical IDs back to actual Gene Names:

```bash
python scripts/merge_results.py
```

Output: `ZIBRA_Final_Results.csv`

## Advanced Customization
* Gene Filtering: Modify `step1_univariate_filter` or the `np.mean` threshold in `scripts/prepare_data.py.`

* Significance Threshold: The decision tree threshold (`alpha = 1e-8`) can be modified directly in `src/zibra_mpi.cpp`.