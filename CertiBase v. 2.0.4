import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

top3_results = []

def isanumber(a):
    """Check if a value can be converted to float"""
    try:
        float(a)
        return True
    except (ValueError, TypeError):
        return False

def calcular_similaridade(query, reference):
    total_similarity = 0
    valid_loci_count = 0

    for q, r in zip(query, reference):
        if isanumber(q) and isanumber(r) and not (np.isnan(float(q)) or np.isnan(float(r))):
            q = float(q)
            r = float(r)
            loco_similarity = max(0, 1 - abs(q - r) / r)
            total_similarity += loco_similarity
            valid_loci_count += 1

    if valid_loci_count == 0:
        return 0

    return total_similarity / valid_loci_count

def load_file(entry_widget):
    """Load Excel file through dialog"""
    filename = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if filename:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, filename)

def calculate_and_display():
    """Main function to calculate and display similarity results"""
    global top3_results
    top3_results = []

    db_filename = entry_database.get().strip()
    query_filename = entry_query.get().strip()

    if not db_filename or not query_filename:
        messagebox.showwarning("Warning", "Please select both files.")
        return

    try:
        # Read Excel files
        db_df = pd.read_excel(db_filename, engine='openpyxl', header=1)
        query_df = pd.read_excel(query_filename, engine='openpyxl', header=1)
    except FileNotFoundError as e:
        messagebox.showerror("File Error", f"File not found:\n{e}")
        return
    except Exception as e:
        messagebox.showerror("Read Error", f"Error reading Excel files:\n{e}")
        return

    # Check if DataFrames are not empty
    if db_df.empty or query_df.empty:
        messagebox.showerror("Error", "One or both files are empty.")
        return

    db_name_col = db_df.columns[0]
    query_name_col = query_df.columns[0]

    results_text.delete(1.0, tk.END)

    # Process each query sample
    for idx_query, query_row in query_df.iterrows():
        query_name = query_row[query_name_col]
        if pd.isna(query_name) or str(query_name).strip() == '':
            continue

        query_data = query_row.drop(query_name_col)
        similarities = {}

        # Calculate similarity with each sample in the database
        for idx_ref, ref_row in db_df.iterrows():
            ref_name = ref_row[db_name_col]
            if pd.isna(ref_name) or str(ref_name).strip() == '':
                continue
            ref_data = ref_row.drop(db_name_col)
            similarity = calcular_similaridade(query_data, ref_data)
            similarities[ref_name] = similarity

        if not similarities:
            continue

        top3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

        results_text.insert(tk.END, f"\nQuery sample: {query_name}\n")
        for ref_name, score in top3:
            results_text.insert(tk.END, f"  ↳ {ref_name}: {score * 100:.2f}%\n")
            top3_results.append((query_name, ref_name, score))

        # Blank separator line
        top3_results.append(("", "", ""))

    # Generate matrix and dendrogram
    try:
        generate_matrix_and_dendrogram(query_df, db_df, query_name_col, db_name_col, bootstrap_var.get())
    except Exception as e:
        results_text.insert(tk.END, f"\nError generating matrix/dendrogram: {e}\n")

def calcular_suporte_bootstrap(data, original_linkage, replicates=100):
    """Calculate bootstrap support for clusters in dendrogram"""
    n = data.shape[0]
    clusters_suporte = defaultdict(int)

    def get_cluster_members(link, n):
        clusters = {}
        for i, (c1, c2, dist, sample_count) in enumerate(link):
            c1, c2 = int(c1), int(c2)
            members = set()
            if c1 < n:
                members.add(c1)
            else:
                members.update(clusters.get(c1, set()))
            if c2 < n:
                members.add(c2)
            else:
                members.update(clusters.get(c2, set()))
            clusters[i + n] = members
        return clusters

    original_clusters = get_cluster_members(original_linkage, n)

    for rep in range(replicates):
        try:
            # Bootstrap sampling of columns
            sampled_cols = np.random.choice(data.shape[1], size=data.shape[1], replace=True)
            data_boot = data.iloc[:, sampled_cols]

            # Calculate bootstrap similarity matrix
            matrix = pd.DataFrame(index=data_boot.index, columns=data_boot.index, dtype=float)
            np.fill_diagonal(matrix.values, 1.0)  # Diagonal = 1
            
            for i1 in range(len(data_boot)):
                for i2 in range(i1 + 1, len(data_boot)):
                    sim = calcular_similaridade(data_boot.iloc[i1], data_boot.iloc[i2])
                    matrix.iat[i1, i2] = sim
                    matrix.iat[i2, i1] = sim

            # Convert to distance matrix
            dist = 1 - matrix.values
            condensed_dist = squareform(dist, checks=False)
            
            linkage_boot = linkage(condensed_dist, method='average')
            bootstrap_clusters = get_cluster_members(linkage_boot, n)

            # Count how many times each original cluster appears in bootstrap
            for cluster_id, members in original_clusters.items():
                if any(members == bmembers for bmembers in bootstrap_clusters.values()):
                    clusters_suporte[cluster_id] += 1

        except Exception:
            # If there's an error in one replicate, continue with others
            continue

    # Convert counts to percentages
    for c in clusters_suporte:
        clusters_suporte[c] = clusters_suporte[c] / replicates * 100

    return clusters_suporte

def plot_dendrogram_with_bootstrap(linkage_matrix, labels, supports, cutoff=50):
    """Plot dendrogram with bootstrap support values"""
    plt.figure(figsize=(12, 8))
    dendro = dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)

    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])
    color_list = dendro['color_list']

    ax = plt.gca()

    # Add bootstrap values to nodes
    for i, d, c in zip(icoord, dcoord, color_list):
        x = 0.5 * (i[1] + i[2])
        y = d[1]

        # Find corresponding cluster
        for cluster_idx, row in enumerate(linkage_matrix):
            dist = row[2]
            if abs(dist - y) < 1e-5:
                support_val = supports.get(cluster_idx + len(labels), 0)
                if support_val >= cutoff:
                    ax.text(x, y, f"{int(round(support_val))}%", 
                           va='bottom', ha='center', fontsize=9, color='red', weight='bold')
                break

    plt.title("CertiBase 2.0 - UPGMA Dendrogram (with bootstrap)", fontsize=14, weight='bold')
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    
    try:
        plt.savefig("CertiBase2.0_dendrogram_upgma_bootstrap.jpg", dpi=300, bbox_inches='tight')
        plt.savefig("CertiBase2.0_dendrogram_upgma_bootstrap.pdf", bbox_inches='tight')
        plt.show()
    except Exception as e:
        results_text.insert(tk.END, f"Error saving dendrogram: {e}\n")

def generate_matrix_and_dendrogram(query_df, db_df, query_name_col, db_name_col, bootstrap_enabled):
    """Generate similarity matrix and dendrogram"""
    # Combine dataframes
    combined_df = pd.concat([query_df, db_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=[query_name_col if query_name_col in combined_df.columns else db_name_col])
    
    if combined_df.empty:
        results_text.insert(tk.END, "Error: No valid samples found.\n")
        return

    names = combined_df.iloc[:, 0].tolist()
    data = combined_df.iloc[:, 1:]

    # Remove samples with empty names
    valid_indices = [i for i, name in enumerate(names) if pd.notna(name) and str(name).strip() != '']
    names = [names[i] for i in valid_indices]
    data = data.iloc[valid_indices]

    if len(names) < 2:
        results_text.insert(tk.END, "Error: At least 2 samples are required for analysis.\n")
        return

    # Create similarity matrix
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)  # Diagonal = 1

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            sim = calcular_similaridade(data.iloc[i], data.iloc[j])
            matrix.iat[i, j] = sim
            matrix.iat[j, i] = sim

    # Save similarity matrix with CertiBase 2.0 title
    matrix_filename = "CertiBase2.0_genetic_similarity_matrix.xlsx"
    try:
        with pd.ExcelWriter(matrix_filename, engine='openpyxl') as writer:
            # Create a title row
            title_df = pd.DataFrame({"": ["CertiBase 2.0"]})
            title_df.to_excel(writer, sheet_name='Similarity Matrix', index=False, header=False)
            
            # Add the matrix data starting from row 2
            matrix.to_excel(writer, sheet_name='Similarity Matrix', startrow=2)
        
        results_text.insert(tk.END, f"\nSimilarity matrix saved as: {matrix_filename}\n")
    except Exception as e:
        results_text.insert(tk.END, f"\nError saving matrix: {e}\n")

    # Convert to distance matrix
    matrix_float = matrix.astype(float)
    distance_matrix = 1 - matrix_float.values
    
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)
    
    try:
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')
    except Exception as e:
        results_text.insert(tk.END, f"Error computing dendrogram: {e}\n")
        return

    # Plot dendrogram
    if bootstrap_enabled:
        results_text.insert(tk.END, "Calculating bootstrap supports (this may take a while)...\n")
        root.update_idletasks()
        
        try:
            supports = calcular_suporte_bootstrap(data, linkage_matrix, replicates=100)
            plot_dendrogram_with_bootstrap(linkage_matrix, names, supports)
            results_text.insert(tk.END, "Dendrogram with bootstrap saved as: CertiBase2.0_dendrogram_upgma_bootstrap.jpg/pdf\n")
        except Exception as e:
            results_text.insert(tk.END, f"Bootstrap error: {e}\n")
            # Plot simple dendrogram as fallback
            plot_simple_dendrogram(linkage_matrix, names)
    else:
        plot_simple_dendrogram(linkage_matrix, names)

def plot_simple_dendrogram(linkage_matrix, names):
    """Plot simple dendrogram without bootstrap"""
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=names, leaf_rotation=90, leaf_font_size=10)
    plt.title("CertiBase 2.0 - UPGMA Dendrogram", fontsize=14, weight='bold')
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    
    try:
        plt.savefig("CertiBase2.0_dendrogram_upgma.jpg", dpi=300, bbox_inches='tight')
        plt.savefig("CertiBase2.0_dendrogram_upgma.pdf", bbox_inches='tight')
        plt.show()
        results_text.insert(tk.END, f"Dendrogram saved as: CertiBase2.0_dendrogram_upgma.jpg and CertiBase2.0_dendrogram_upgma.pdf\n")
    except Exception as e:
        results_text.insert(tk.END, f"Error saving dendrogram: {e}\n")

def save_results():
    """Save Top 3 results to TSV and PDF"""
    if not top3_results:
        messagebox.showinfo("No data", "No results available to save.")
        return

    df_results = pd.DataFrame(top3_results, columns=["Query", "Reference", "Similarity"])
    
    # Convert similarity to percentage, handling empty values
    df_results["Similarity (%)"] = pd.to_numeric(df_results["Similarity"], errors='coerce') * 100
    df_results.drop(columns=["Similarity"], inplace=True)

    save_path = filedialog.asksaveasfilename(
        defaultextension=".tsv",
        filetypes=[("TSV file", "*.tsv"), ("All files", "*.*")]
    )
    if not save_path:
        return

    try:
        # Add CertiBase 2.0 title to TSV
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("CertiBase 2.0\n\n")
            df_results.to_csv(f, sep="\t", index=False)
        
        results_text.insert(tk.END, f"\nTop 3 similarity results saved as TSV: {save_path}\n")
    except Exception as e:
        results_text.insert(tk.END, f"Error saving TSV: {e}\n")
        return

    # Save PDF with CertiBase 2.0 title
    pdf_path = save_path.rsplit('.', 1)[0] + ".pdf"
    try:
        with PdfPages(pdf_path) as pdf:
            lines_per_page = 45
            lines = ["CertiBase 2.0", ""]  # Add title and blank line
            
            for idx, row in df_results.iterrows():
                if pd.isna(row["Query"]) or str(row["Query"]).strip() == "":
                    lines.append("")  # blank line
                    continue
                line = f"{row['Query']} ↔ {row['Reference']}: {row['Similarity (%)']:.2f}%"
                lines.append(line)

            # Create PDF pages
            for i in range(0, len(lines), lines_per_page):
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                page_text = "\n".join(lines[i:i + lines_per_page])
                ax.text(0, 1, page_text, va='top', fontsize=10, family='monospace')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        results_text.insert(tk.END, f"Top 3 similarity results saved as PDF: {pdf_path}\n")
    except Exception as e:
        results_text.insert(tk.END, f"Error saving PDF: {e}\n")

def calculate_genetic_indices():
    """Calculate genetic indices for query samples"""
    query_filename = entry_query.get().strip()

    if not query_filename:
        messagebox.showwarning("Warning", "Please select the query samples file.")
        return

    try:
        query_df = pd.read_excel(query_filename, engine='openpyxl', header=1)
    except Exception as e:
        messagebox.showerror("Read Error", f"Error reading query Excel file:\n{e}")
        return

    if query_df.empty:
        messagebox.showerror("Error", "The query file is empty.")
        return

    query_name_col = query_df.columns[0]
    query_data = query_df.drop(columns=[query_name_col])

    A_list = []
    Ae_list = []
    He_list = []
    Ho_list = []
    F_list = []
    samples = []

    for idx, row in query_data.iterrows():
        sample = query_df.iloc[idx, 0]
        if pd.isna(sample) or str(sample).strip() == '':
            continue
            
        samples.append(sample)

        # Filter only valid numeric values > 0 (excluding missing alleles)
        alleles = []
        for val in row:
            if isanumber(val):
                num_val = float(val)
                if not np.isnan(num_val) and num_val > 0:  # Only consider present alleles
                    alleles.append(num_val)
        
        if len(alleles) == 0:
            # If no valid alleles, set default values
            A_list.append(0)
            Ae_list.append(0)
            He_list.append(0)
            Ho_list.append(0)
            F_list.append(0)
            continue

        # Convert to numpy array
        alleles = np.array(alleles)
        
        # Number of different alleles (A)
        unique_alleles = np.unique(alleles)
        A = len(unique_alleles)

        # Calculate allele frequencies
        allele_counts = {}
        for allele in alleles:
            allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        total_alleles = len(alleles)
        frequencies = np.array([count / total_alleles for count in allele_counts.values()])

        # Effective number of alleles Ae = 1 / (sum p_i^2)
        p_squared_sum = np.sum(frequencies ** 2)
        Ae = 1 / p_squared_sum if p_squared_sum > 0 else 0

        # Expected heterozygosity He = 1 - sum p_i^2
        He = 1 - p_squared_sum

        # Observed heterozygosity Ho (simplified approximation)
        # Assuming each locus can be heterozygous if multiple alleles are present
        # This is a simplified calculation - ideally you'd need genotype data
        if A > 1:
            Ho = 1 - max(frequencies)  # 1 - frequency of most common allele
        else:
            Ho = 0

        # Fixation index F = (He - Ho) / He
        F = (He - Ho) / He if He > 0 else 0

        # Ensure Ae <= A (biological constraint)
        if Ae > A:
            Ae = A

        A_list.append(round(A, 3))
        Ae_list.append(round(Ae, 3))
        He_list.append(round(He, 3))
        Ho_list.append(round(Ho, 3))
        F_list.append(round(F, 3))

    if not samples:
        messagebox.showwarning("Warning", "No valid samples found.")
        return

    # Calculate means for Query samples
    mean_A = round(np.mean(A_list), 3) if A_list else 0
    mean_Ae = round(np.mean(Ae_list), 3) if Ae_list else 0
    mean_He = round(np.mean(He_list), 3) if He_list else 0
    mean_Ho = round(np.mean(Ho_list), 3) if Ho_list else 0
    mean_F = round(np.mean(F_list), 3) if F_list else 0

    # Add mean row to the data
    samples.append("Mean")
    A_list.append(mean_A)
    Ae_list.append(mean_Ae)
    He_list.append(mean_He)
    Ho_list.append(mean_Ho)
    F_list.append(mean_F)

    # Create DataFrame with simplified headers
    df_indices = pd.DataFrame({
        "Sample": samples,
        "A": A_list,
        "Ae": Ae_list,
        "He": He_list,
        "Ho": Ho_list,
        "F": F_list,
    })

    tsv_path = "CertiBase2.0_genetic_indices_query.tsv"
    pdf_path = "CertiBase2.0_genetic_indices_query.pdf"

    try:
        # Save TSV with CertiBase 2.0 title
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("CertiBase 2.0\n\n")
            df_indices.to_csv(f, sep="\t", index=False)
    except Exception as e:
        messagebox.showerror("Save Error", f"Error saving TSV:\n{e}")
        return

    try:
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('tight')
            ax.axis('off')
            
            # Add title
            ax.text(0.5, 0.95, 'CertiBase 2.0', transform=ax.transAxes, 
                   fontsize=16, weight='bold', ha='center')
            
            table = ax.table(cellText=df_indices.values,
                           colLabels=df_indices.columns,
                           loc='center',
                           cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Highlight the mean row
            mean_row_idx = len(df_indices) - 1
            for j in range(len(df_indices.columns)):
                table[(mean_row_idx + 1, j)].set_facecolor('#E6E6FA')
                table[(mean_row_idx + 1, j)].set_text_props(weight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        messagebox.showerror("Save Error", f"Error saving PDF:\n{e}")
        return

    messagebox.showinfo("Success", f"Genetic indices saved as TSV and PDF:\n{tsv_path}\n{pdf_path}")

# Graphical User Interface
root = tk.Tk()
root.title("CertiBase 2.0")
root.geometry("1000x600")
root.configure(bg="#f0f0f0")

# Title label
title_label = tk.Label(root, text="CertiBase 2.0", font=("Arial", 24, "bold"), 
                      bg="#f0f0f0", fg="#2c3e50")
title_label.pack(pady=(10, 5))

# Top frame for file selection
frame_top = tk.Frame(root, bg="#f0f0f0")
frame_top.pack(pady=10)

label_database = tk.Label(frame_top, text="Database (reference) Excel:", bg="#f0f0f0", font=("Arial", 14))
label_database.grid(row=0, column=0, sticky="w", padx=5)
entry_database = tk.Entry(frame_top, width=50)
entry_database.grid(row=0, column=1, padx=5)
btn_browse_db = tk.Button(frame_top, text="Browse", command=lambda: load_file(entry_database), 
                         bg="#4caf50", fg="white", width=12, height=2, font=("Arial", 14))
btn_browse_db.grid(row=0, column=2, padx=5)

label_query = tk.Label(frame_top, text="Query samples Excel:", bg="#f0f0f0", font=("Arial", 14))
label_query.grid(row=1, column=0, sticky="w", padx=5)
entry_query = tk.Entry(frame_top, width=50)
entry_query.grid(row=1, column=1, padx=5)
btn_browse_query = tk.Button(frame_top, text="Browse", command=lambda: load_file(entry_query), 
                           bg="#2196f3", fg="white", width=12, height=2, font=("Arial", 14))
btn_browse_query.grid(row=1, column=2, padx=5)

bootstrap_var = tk.BooleanVar()
check_bootstrap = tk.Checkbutton(frame_top, text="Use bootstrap", variable=bootstrap_var, bg="#f0f0f0", font=("Arial", 14))
check_bootstrap.grid(row=2, column=1, sticky="w", pady=5)

# Frame for buttons
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=10)

btn_top3 = tk.Button(frame_buttons, text="Generate similarity matrix + Dendrogram", 
                    command=calculate_and_display, bg="#e91e63", fg="white", width=40, height=2, font=("Arial", 14))
btn_top3.grid(row=0, column=0, padx=5, pady=5)

btn_save = tk.Button(frame_buttons, text="Save Top 3 results (TSV + PDF)", 
                    command=save_results, bg="#ff9800", fg="white", width=40, height=2, font=("Arial", 14))
btn_save.grid(row=0, column=1, padx=5, pady=5)

btn_genetic_indices = tk.Button(frame_buttons, text="Genetic indexes of the Query (A, Ae, He, Ho, F)", 
                              command=calculate_genetic_indices, bg="#0000ff", fg="white", width=50, height=2, font=("Arial", 14))
btn_genetic_indices.grid(row=1, column=0, columnspan=2, pady=10)

# Frame for results
results_frame = tk.Frame(root, bg="#f0f0f0")
results_frame.pack(fill="both", expand=True, padx=10, pady=10)

scrollbar = tk.Scrollbar(results_frame)
scrollbar.pack(side="right", fill="y")

results_text = tk.Text(results_frame, yscrollcommand=scrollbar.set, bg="white", fg="black")
results_text.pack(fill="both", expand=True)
scrollbar.config(command=results_text.yview)

# Start the application
if __name__ == "__main__":
    root.mainloop()
