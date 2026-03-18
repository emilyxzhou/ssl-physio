import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
paths = [
    os.path.join(
        USER_ROOT, "ssl-physio", "src"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "dataloaders"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "mamba"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "s4_models"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "trainers"
    )
]
for path in paths:
    sys.path.insert(0, path)
physio_data_path = os.path.join(
    USER_ROOT, "physio-data", "src"
)
sys.path.append(physio_data_path)

import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from tiles_dataloader import get_embeddings_from_file, TilesDataset, generate_binary_labels, generate_continuous_labels_day


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Line settings
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        
        # Legend settings
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
    })


if __name__ == "__main__":
    setup_plot_style()
    method = "raw"
    binary_labels = ["age", "shift", "anxiety", "stress"]
    legend_dict = {
        "age": ["< 40 years", ">= 40 years"],
        "shift": ["Day shift", "Night shift"],
        "anxiety": ["Low anxiety", "High anxiety"],
        "stress": ["Low stress", "High stress"]
    }
    markers = {0: "o", 1: "^"}
    model_types = ["s4", "mamba"]
    # mask_pcts = [10, 30, 50, 70]
    mask_pcts = [50]

    for model_type in model_types:
        for mask_pct in mask_pcts:
            print("="*50)
            print(f"{model_type} {mask_pct}% masking")
            print("="*50)
            
            df = {
                "ID": None,
                "shift": None,
                "age": None,
                "anxiety": None,
                "stress": None
            }

            subject_ids, dates, embeddings = get_embeddings_from_file(model_type, mask_pct, method=method)
            subject_ids = np.asarray(subject_ids)
            df["ID"] = subject_ids

            embeddings = np.asarray(embeddings)
            if len(embeddings.shape) > 2: embeddings = np.mean(embeddings, axis=1)
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            for label_type in binary_labels:
                labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
                labels = np.asarray(labels)
                df[label_type] = labels

            df = pd.DataFrame(df)

            tsne = TSNE(
                n_components=2, 
                perplexity=40, 
                init='pca', 
                learning_rate='auto', 
                random_state=13,
                n_jobs=-1 # Uses all CPU cores for speed
            )
            tsne_results = tsne.fit_transform(embeddings_scaled)

            df['tsne_1'] = tsne_results[:, 0]
            df['tsne_2'] = tsne_results[:, 1]

            # Plot binary features
            fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
            axes1 = axes1.flatten()
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
            axes2 = axes2.flatten()

            for i, col in enumerate(binary_labels):
                # Plot by binary label --------------------------------------------------
                sns.scatterplot(
                    data=df[['tsne_1', 'tsne_2', col]], x='tsne_1', y='tsne_2', hue=col,
                    ax=axes1[i], palette='coolwarm', alpha=0.7, s=40
                )
                axes1[i].set_xticks([])
                axes1[i].set_yticks([])
                axes1[i].set_xlabel('')
                axes1[i].set_ylabel('')
                handles, labels = axes1[i].get_legend_handles_labels()
    
                new_handles = []
                new_labels = []
                for h, l in zip(handles, labels):
                    if l in ['0', '1']:
                        # Force legend icon color to black/grey so it doesn't look like a specific ID
                        h.set_color('black')
                        new_labels.append(f"{legend_dict[col][int(l)]}")
                        new_handles.append(h)
                
                axes1[i].legend(
                    handles=new_handles, 
                    labels=new_labels, 
                    loc='upper right',
                    frameon=True
                )

                axes1[i].set_title(f't-SNE by {col.capitalize()}', fontsize=18)
                fig1.suptitle(f'{model_type.capitalize()} ({mask_pct}%)', fontsize=20, fontweight='bold')

                # Plot by ID and binary label --------------------------------------------------
                sns.scatterplot(
                    data=df,
                    x='tsne_1',
                    y='tsne_2',
                    hue='ID',       # Color by subject ID
                    style=col,      # Shape by binary label
                    markers=markers,
                    ax=axes2[i],
                    palette='viridis',
                    legend='full',
                    alpha=0.7
                )   
                axes2[i].set_xticks([])
                axes2[i].set_yticks([])
                axes2[i].set_xlabel('')
                axes2[i].set_ylabel('')
                handles, labels = axes2[i].get_legend_handles_labels()
    
                new_handles = []
                new_labels = []
                for h, l in zip(handles, labels):
                    if l in ['0', '1']:
                        # Force legend icon color to black/grey so it doesn't look like a specific ID
                        h.set_color('black')
                        new_labels.append(f"{legend_dict[col][int(l)]}")
                        new_handles.append(h)
                
                axes2[i].legend(
                    handles=new_handles, 
                    labels=new_labels, 
                    loc='upper right',
                    frameon=True
                )

                axes2[i].set_title(f't-SNE by {col.capitalize()}', fontsize=14)
                fig2.suptitle(f'{model_type.capitalize()} ({mask_pct}%)', fontsize=15, fontweight='bold')
                # axes2[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')


            # Plot subject ID
            fig3, axes3 = plt.subplots(figsize=(10, 10))
            sns.scatterplot(
                data=df, x='tsne_1', y='tsne_2', hue='ID',
                ax=axes3, palette='husl', alpha=0.7, s=40,
                legend=False
            )
            axes3.set_xticks([])
            axes3.set_yticks([])
            axes3.set_xlabel('')
            axes3.set_ylabel('')

            axes3.set_title('t-SNE: Subject ID', fontsize=18, fontweight='bold')
            
            fig1.tight_layout()
            fig1.savefig(f"/home/emilyzho/ssl-physio/plots/embeddings/tsne_{model_type}_{mask_pct}.png", dpi=300, bbox_inches='tight', pad_inches=0.2)

            fig2.tight_layout()
            fig2.savefig(f"/home/emilyzho/ssl-physio/plots/embeddings/tsne_{model_type}_{mask_pct}_subject_split.png", dpi=300, bbox_inches='tight', pad_inches=0.2)

            fig3.tight_layout()
            fig3.savefig(f"/home/emilyzho/ssl-physio/plots/embeddings/tsne_{model_type}_{mask_pct}_subjects.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.close()
