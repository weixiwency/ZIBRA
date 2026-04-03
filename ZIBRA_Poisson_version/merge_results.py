import pandas as pd
import glob
import os
import shutil

def main():
    chunk_dir = "output_chunks"
    output_file = "ZIBRA_Final_Results_Poisson.csv"
    
    print("1. Loading gene names...")
    gene_map = {}
    if os.path.exists("gene_names.txt"):
        with open("gene_names.txt", "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2: gene_map[int(parts[0])] = parts[1]
    
    print("2. Merging chunks from folder...")
    csv_files = glob.glob(f"{chunk_dir}/results_rank_*.csv")
    if not csv_files:
        print(f"Error: No files found in {chunk_dir}")
        return

    # 分块读取合并，防止内存溢出
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f))
    
    df_master = pd.concat(df_list, ignore_index=True)

    print("3. Finalizing Master Table...")
    df_master.insert(0, 'Gene_A', df_master['idx_A'].map(gene_map))
    df_master.insert(1, 'Gene_B', df_master['idx_B'].map(gene_map))
    df_master.drop(columns=['idx_A', 'idx_B'], inplace=True)

    df_master.to_csv(output_file, index=False)
    print(f"Success! Final table saved as {output_file}")
    
    # 询问是否清理
    cleanup = input("Do you want to delete the output_chunks folder? (y/n): ")
    if cleanup.lower() == 'y':
        shutil.rmtree(chunk_dir)
        print("Cleanup finished.")

if __name__ == "__main__":
    main()