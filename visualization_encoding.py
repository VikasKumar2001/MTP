import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class EncodingVisualizer:
    """
    Thesis-Level Visualization Engine for Phase 1: Encoding & Compression.
    Automatically generates 11 structured directories per CAD model.
    """
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid")
        
        self.geom_labels = [
            "Norm Vol", "Norm Area", "Curvature", 
            "Dim X", "Dim Y", "Dim Z", 
            "Asp XY", "Asp XZ", "Asp YZ", 
            "PCA 1", "PCA 2", "PCA 3", "Pad", "Pad", "Pad", "Pad"
        ]

    def create_model_folders(self, model_id: str) -> dict:
        base_path = os.path.join(self.output_dir, model_id, "visualizations")
        
        folders = {
            "input": os.path.join(base_path, "01_input"),
            "slicing": os.path.join(base_path, "02_slicing"),
            "geometry": os.path.join(base_path, "03_geometry"),
            "shapedna": os.path.join(base_path, "04_shapedna"),
            "fpfh": os.path.join(base_path, "05_fpfh"),
            "pointnet": os.path.join(base_path, "06_pointnet"),
            "dinov2": os.path.join(base_path, "07_dinov2"),
            "fusion": os.path.join(base_path, "08_feature_fusion"),
            "qr": os.path.join(base_path, "09_qr_encoding"),
            "compression": os.path.join(base_path, "10_compression"),
            "performance": os.path.join(base_path, "11_performance")
        }
        
        for path in folders.values():
            os.makedirs(path, exist_ok=True)
            
        return folders

    # --- 01 to 05 (Input, Slicing, Geometry, ShapeDNA, FPFH) ---
    def plot_input_statistics(self, summary: dict, folders: dict):
        if not summary: return
        stats = {
            "Volume": summary.get("volume", 0),
            "Surface Area": summary.get("surface_area", 0),
            "Triangles": summary.get("triangle_count", 0),
            "Vertices": summary.get("vertex_count", 0)
        }
        plt.figure(figsize=(8, 5))
        plt.bar(stats.keys(), stats.values(), color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        plt.yscale('log')
        plt.title("CAD Model Raw Statistics (Log Scale)")
        plt.ylabel("Count / Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(folders["input"], "geometry_statistics.png"))
        plt.close()

    def plot_slice_analysis(self, slice_result: dict, folders: dict):
        if not slice_result or "layers" not in slice_result: return
        z_vals = [lyr["z"] for lyr in slice_result["layers"] if lyr]
        areas = [lyr["area"] for lyr in slice_result["layers"] if lyr]
        
        plt.figure(figsize=(10, 4))
        plt.plot(z_vals, areas, color='purple', linewidth=2)
        plt.fill_between(z_vals, areas, alpha=0.3, color='purple')
        plt.title("Cross-Sectional Area vs Z-Height")
        plt.xlabel("Z Height")
        plt.ylabel("Area")
        plt.tight_layout()
        plt.savefig(os.path.join(folders["slicing"], "cross_section_area.png"))
        plt.close()

        signature = slice_result.get("slice_signature", [])
        if signature:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(signature)), signature, color='indigo')
            plt.title("Normalized 20-D Slice Signature")
            plt.tight_layout()
            plt.savefig(os.path.join(folders["slicing"], "slice_signature.png"))
            plt.close()

    def plot_geometry_features(self, geom_features: dict, folders: dict):
        vector = geom_features.get("geometry_vector", [])
        if not vector or len(vector) != len(self.geom_labels): return
        plt.figure(figsize=(12, 5))
        sns.barplot(x=self.geom_labels, y=vector, hue=self.geom_labels, palette="viridis", legend=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Normalized Geometric Features (16-D)")
        plt.tight_layout()
        plt.savefig(os.path.join(folders["geometry"], "normalized_geometry_features.png"))
        plt.close()

    def plot_shapedna(self, shapedna_features: dict, folders: dict):
        vector = shapedna_features.get("shape_dna_vector", [])
        if not vector: return
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(vector)), vector, marker='o', markersize=4, linestyle='-', color='crimson')
        plt.title("ShapeDNA Eigenvalue Spectrum (First 64 Frequencies)")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(folders["shapedna"], "eigenvalue_spectrum.png"))
        plt.close()

    def plot_fpfh(self, fpfh_features: dict, folders: dict):
        vector = fpfh_features.get("fpfh_vector", [])
        if not vector: return
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(vector)), vector, color='teal')
        plt.title("FPFH Global Descriptor (33-D)")
        plt.xlabel("Histogram Bin")
        plt.ylabel("Normalized Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(folders["fpfh"], "fpfh_histogram.png"))
        plt.close()

    def plot_pointnet(self, pointnet_features: dict, trimesh_model, folders: dict):
        vector = pointnet_features.get("pointnet_vector", [])
        if not vector: return
        save_dir = folders["pointnet"]
        
        plt.figure(figsize=(16, 4))
        sns.heatmap(np.array(vector).reshape(1, -1), cmap="viridis", cbar=True, yticklabels=False)
        plt.title("PointNet 1024-D Feature Embedding Heatmap")
        plt.xlabel("Feature Dimension")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_heatmap_1024d.png"))
        plt.close()

        plt.figure(figsize=(14, 5))
        plt.plot(range(1, len(vector) + 1), vector, linewidth=1, color='darkcyan')
        plt.xlabel("PointNet Feature Dimension")
        plt.ylabel("Feature Value")
        plt.title("PointNet 1024-D Feature Embedding Line Plot")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_lineplot_1024d.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(vector, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.title("PointNet Feature Value Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_distribution.png"))
        plt.close()

    # --- 07. DINOv2 Analysis ---
    def plot_dinov2(self, dinov2_features: dict, rendered_views: list, folders: dict):
        vector = dinov2_features.get("dinov2_vector", [])
        if not vector: return
        save_dir = folders["dinov2"]

        # A. Heatmap
        plt.figure(figsize=(16, 4))
        sns.heatmap(np.array(vector).reshape(1, -1), cmap="viridis", cbar=True, yticklabels=[])
        plt.title("DINOv2 768-D Feature Embedding Heatmap")
        plt.xlabel("Feature Dimension")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_heatmap_768d.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # B. Line Plot
        plt.figure(figsize=(14, 5))
        plt.plot(range(1, len(vector) + 1), vector, linewidth=1, color='darkorange')
        plt.xlabel("DINOv2 Feature Dimension")
        plt.ylabel("Feature Value")
        plt.title("DINOv2 768-D Feature Profile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_lineplot_768d.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # C. Distribution
        plt.figure(figsize=(8, 5))
        plt.hist(vector, bins=40, color='peru', edgecolor='black', alpha=0.7)
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.title("DINOv2 Feature Value Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # D. Rendered Views Grid
        if rendered_views and len(rendered_views) == 12:
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i, ax in enumerate(axes.flat):
                ax.imshow(rendered_views[i])
                ax.axis('off')
                ax.set_title(f"View {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "rendered_views.png"), dpi=300, bbox_inches="tight")
            plt.close()

        # E. View-wise Heatmap
        view_embeddings = dinov2_features.get("dinov2_view_embeddings", [])
        if view_embeddings and len(view_embeddings) == 12:
            plt.figure(figsize=(16, 6))
            sns.heatmap(np.array(view_embeddings), cmap="viridis", center=0, cbar=True, 
                        yticklabels=[f"View {i+1}" for i in range(12)])
            plt.xlabel("DINOv2 Feature Dimension")
            plt.ylabel("Rendered View")
            plt.title("View-wise DINOv2 Feature Embeddings")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "view_embedding_comparison.png"), dpi=300, bbox_inches="tight")
            plt.close()

    # --- 08. Feature Fusion Analysis ---
    def plot_fusion(self, all_features: dict, latent_vector: list, folders: dict):
        if not latent_vector: return
        save_dir = folders["fusion"]

        # A. Heatmap
        plt.figure(figsize=(16, 4))
        sns.heatmap(np.array(latent_vector).reshape(1, -1), cmap="viridis", center=0, cbar=True, yticklabels=[])
        plt.title("Fused 512-D Latent Representation Heatmap")
        plt.xlabel("Latent Dimension")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_heatmap_512d.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # B. Line Plot
        plt.figure(figsize=(14, 5))
        plt.plot(range(1, len(latent_vector) + 1), latent_vector, linewidth=1, color='indigo')
        plt.xlabel("Latent Dimension")
        plt.ylabel("Latent Value")
        plt.title("512-D Fused Latent Feature Profile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_lineplot_512d.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # C. Distribution
        plt.figure(figsize=(8, 5))
        plt.hist(latent_vector, bins=40, color='mediumpurple', edgecolor='black', alpha=0.7)
        plt.xlabel("Latent Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of 512-D Latent Representation")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # D. Feature Dimensionality Bar Chart
        feature_names = ["Geometry", "Slice", "ShapeDNA", "FPFH", "PointNet", "DINOv2", "Fusion"]
        dimensions = [
            len(all_features.get("geometry_vector", [])),
            len(all_features.get("slice_vector", [])),
            len(all_features.get("shape_dna_vector", [])),
            len(all_features.get("fpfh_vector", [])),
            len(all_features.get("pointnet_vector", [])),
            len(all_features.get("dinov2_vector", [])),
            len(latent_vector)
        ]

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, dimensions, color='slategray')
        plt.xlabel("Feature Dimension")
        plt.title("Dimensionality of Multi-Descriptor Representation")
        
        for index, value in enumerate(dimensions):
            plt.text(value, index, f' {value}')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_contribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # --- Dataset-Level Visualizations ---
    def plot_dataset_tsne(self, latent_vectors: list, labels: list, save_dir: str):
        if len(latent_vectors) < 2: return
        os.makedirs(save_dir, exist_ok=True)
        perplexity = min(30, len(latent_vectors) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(np.array(latent_vectors))
        classes = [lbl.split('_')[0] if '_' in lbl else lbl for lbl in labels]
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=classes, palette="tab10", s=100)
        plt.title("t-SNE Projection of 512-D Latent Space")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "tsne.png"))
        plt.close()

    def plot_dataset_pca(self, latent_vectors: list, labels: list, save_dir: str):
        if len(latent_vectors) < 2: return
        os.makedirs(save_dir, exist_ok=True)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(np.array(latent_vectors))
        classes = [lbl.split('_')[0] if '_' in lbl else lbl for lbl in labels]
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=classes, palette="Set2", s=100)
        plt.title(f"PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca.png"))
        plt.close()

    def plot_dataset_pointnet_pca(self, pointnet_vectors: list, labels: list, save_dir: str):
        """
        PCA visualization of PointNet features across the dataset.
        """
        if len(pointnet_vectors) < 2:
            return

        os.makedirs(save_dir, exist_ok=True)
        X = np.asarray(pointnet_vectors, dtype=np.float32)

        # PCA requires at least 2 dimensions
        if X.shape[1] < 2:
            return

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(10, 7))

        # Use labels as categories (extract base classes like "chair" if formatted as "chair_01")
        classes = [lbl.split('_')[0] if '_' in lbl else lbl for lbl in labels]
        unique_classes = list(dict.fromkeys(classes))

        for cls in unique_classes:
            indices = [i for i, x in enumerate(classes) if x == cls]
            plt.scatter(
                X_2d[indices, 0],
                X_2d[indices, 1],
                s=100,
                label=cls
            )

        plt.title("PCA Projection of PointNet Features")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_path = os.path.join(save_dir, "pointnet_pca.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[VISUALIZATION] PointNet PCA saved: {save_path}")