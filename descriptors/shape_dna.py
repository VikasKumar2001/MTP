import numpy as np
import scipy.sparse.linalg as sla

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


class ShapeDNADescriptor:
    """
    ShapeDNA / Laplace-Beltrami Spectrum Descriptor.

    Extracts a scale-invariant 64-D ShapeDNA representation.

    Pipeline:
        STL Mesh
            ↓
        Mesh Cleaning
            ↓
        Cotangent Laplacian
            ↓
        Barycentric Mass Matrix
            ↓
        Generalized Eigenvalue Problem
            ↓
        Laplace-Beltrami Spectrum
            ↓
        Area Normalization
            ↓
        64-D ShapeDNA
    """

    def __init__(self, num_eigenvalues: int = 64):
        self.num_eigenvalues = num_eigenvalues

    # ==========================================================
    # Mesh Cleaning
    # ==========================================================

    def clean_mesh(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        # ------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3)")

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces must have shape (M, 3)")

        # Remove NaN / Inf vertices
        valid_vertices = np.all(np.isfinite(vertices), axis=1)
        old_to_new = -np.ones(len(vertices), dtype=np.int64)
        valid_indices = np.where(valid_vertices)[0]
        old_to_new[valid_indices] = np.arange(len(valid_indices))
        vertices = vertices[valid_vertices]

        # Remove faces referring to invalid vertices
        valid_faces = np.all(faces >= 0, axis=1) & np.all(faces < len(old_to_new), axis=1)
        faces = faces[valid_faces]

        if len(faces) == 0:
            raise ValueError("No valid faces remain after cleaning.")

        # Remap vertex indices
        faces = old_to_new[faces]

        # ------------------------------------------------------
        # Remove faces with repeated vertices
        # ------------------------------------------------------
        non_repeated = (
            (faces[:, 0] != faces[:, 1]) &
            (faces[:, 1] != faces[:, 2]) &
            (faces[:, 0] != faces[:, 2])
        )
        faces = faces[non_repeated]

        if len(faces) == 0:
            raise ValueError("No non-degenerate faces remain.")

        # ------------------------------------------------------
        # Remove duplicate faces
        # ------------------------------------------------------
        sorted_faces = np.sort(faces, axis=1)
        _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
        faces = faces[np.sort(unique_indices)]

        # ------------------------------------------------------
        # Remove zero-area / extremely tiny triangles
        # ------------------------------------------------------
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_product = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross_product, axis=1) * 0.5

        # Relative threshold
        max_area = np.max(areas) if len(areas) > 0 else 0.0

        if max_area <= 0:
            raise ValueError("Mesh contains no positive-area triangles.")

        area_threshold = max(max_area * 1e-12, 1e-15)
        valid_area = areas > area_threshold
        faces = faces[valid_area]

        if len(faces) == 0:
            raise ValueError("All faces were removed because they have zero or negligible area.")

        # ------------------------------------------------------
        # Remove unreferenced vertices
        # ------------------------------------------------------
        used_vertices = np.unique(faces.reshape(-1))
        new_vertices = vertices[used_vertices]

        remap = -np.ones(len(vertices), dtype=np.int64)
        remap[used_vertices] = np.arange(len(used_vertices))
        new_faces = remap[faces]

        vertices = new_vertices
        faces = new_faces

        return vertices, faces

    # ==========================================================
    # ShapeDNA Extraction
    # ==========================================================

    def extract(self, model_data: dict) -> dict:
        v = np.array(model_data.get("vertices", []), dtype=np.float64)
        f = np.array(model_data.get("faces", []), dtype=np.int64)

        # ------------------------------------------------------
        # Check dependencies
        # ------------------------------------------------------
        if not HAS_IGL:
            print("[WARNING] libigl not found. Returning zero vector.")
            return {"shape_dna_vector": [0.0] * self.num_eigenvalues}

        # ------------------------------------------------------
        # Check mesh
        # ------------------------------------------------------
        if len(v) == 0 or len(f) == 0:
            print("[WARNING] Empty mesh. Returning zero ShapeDNA.")
            return {"shape_dna_vector": [0.0] * self.num_eigenvalues}

        try:
            # ==================================================
            # 1. Clean Mesh
            # ==================================================
            original_vertices = len(v)
            original_faces = len(f)
            v, f = self.clean_mesh(v, f)
            print(f"[ShapeDNA] Mesh cleaned: {original_vertices} → {len(v)} vertices, {original_faces} → {len(f)} faces")

            if len(v) <= 2:
                raise ValueError("Mesh has too few vertices for ShapeDNA.")

            # ==================================================
            # 2. Cotangent Laplacian
            # ==================================================
            L = -igl.cotmatrix(v, f)

            # ==================================================
            # 3. Barycentric Mass Matrix
            # ==================================================
            M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)

            # ==================================================
            # 4. Validate Mass Matrix
            # ==================================================
            mass_diagonal = M.diagonal()

            if len(mass_diagonal) == 0:
                raise ValueError("Mass matrix is empty.")
            if not np.all(np.isfinite(mass_diagonal)):
                raise ValueError("Mass matrix contains NaN or Inf.")
            
            min_mass = np.min(mass_diagonal)
            if min_mass <= 0:
                raise ValueError(f"Mass matrix is singular or contains non-positive entries. Minimum mass = {min_mass}")

            # ==================================================
            # 5. Calculate Surface Area
            # ==================================================
            area = float(np.sum(igl.doublearea(v, f)) * 0.5)
            if area <= 0:
                raise ValueError("Mesh surface area is zero.")

            # ==================================================
            # 6. Number of Eigenvalues
            # ==================================================
            k = min(self.num_eigenvalues + 1, len(v) - 1)
            if k < 2:
                raise ValueError("Not enough vertices to calculate ShapeDNA.")

            # ==================================================
            # 7. Solve Generalized Eigenvalue Problem
            # ==================================================
            try:
                eigenvalues, _ = sla.eigsh(L, k=k, M=M, sigma=1e-8, which="LM")
            except Exception as eigen_error:
                print("[ShapeDNA] Shift-invert solver failed.")
                print(f"[ShapeDNA] Reason: {eigen_error}")
                # Retry without shift-invert
                eigenvalues, _ = sla.eigsh(L, k=k, M=M, which="SM")

            # ==================================================
            # 8. Sort Eigenvalues
            # ==================================================
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)

            # Remove numerical negative zero
            eigenvalues[np.abs(eigenvalues) < 1e-10] = 0.0

            # ==================================================
            # 9. Filter Zero Eigenvalues (Multi-body safe)
            # ==================================================
            if len(eigenvalues) > 0:
                # Drop all eigenvalues that are effectively zero to handle disconnected components
                shape_dna = eigenvalues[eigenvalues > 1e-7]
            else:
                shape_dna = np.array([])

            # Keep requested dimensions
            shape_dna = shape_dna[:self.num_eigenvalues]

            # ==================================================
            # 10. Scale Invariance
            # ==================================================
            shape_dna = shape_dna * area

            # ==================================================
            # 11. Pad if necessary
            # ==================================================
            result = list(shape_dna)
            while len(result) < self.num_eigenvalues:
                result.append(0.0)

            # ==================================================
            # 12. Numerical Cleanup
            # ==================================================
            result = np.asarray(result, dtype=np.float64)
            result[~np.isfinite(result)] = 0.0
            result[np.abs(result) < 1e-10] = 0.0

            # ==================================================
            # SUCCESS
            # ==================================================
            print(f"[ShapeDNA] Successfully extracted {len(result)} eigenvalues.")
            return {
                "shape_dna_vector": [round(float(value), 6) for value in result]
            }

        except Exception as e:
            print(f"[ERROR] ShapeDNA computation failed: {e}")
            return {
                "shape_dna_vector": [0.0] * self.num_eigenvalues
            }