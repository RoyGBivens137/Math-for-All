import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def generate_data():
    np.random.seed(42)
    X = np.random.multivariate_normal(mean=[-8, -4], cov=[[8, 1.0], [1.0, 2.0]], size=100)
    Y = np.random.multivariate_normal(mean=[7, 7], cov=[[1.5, 0.5], [0.5, 1]], size=100)
    return X, Y

def moment_match(X, Y, epsilon=1e-6):
    mu_X, mu_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    X_centered = X - mu_X
    cov_X = np.cov(X_centered.T)
    cov_Y = np.cov((Y - mu_Y).T)
    Ux, Sx, _ = np.linalg.svd(cov_X)
    Uy, Sy, _ = np.linalg.svd(cov_Y)
    Sx_reg = np.maximum(Sx, epsilon)
    T = Uy @ np.diag(np.sqrt(Sy / Sx_reg)) @ Ux.T
    return (X_centered @ T.T) + mu_Y

from scipy.spatial import ConvexHull

def compute_clipped_voronoi(Y, X_all, buffer_ratio=0.1):
    hull = ConvexHull(X_all)
    hull_pts = X_all[hull.vertices]
    min_xy = np.min(hull_pts, axis=0)
    max_xy = np.max(hull_pts, axis=0)
    buffer = buffer_ratio * (max_xy - min_xy)
    min_xy -= buffer
    max_xy += buffer

    box_points = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])
    extended_Y = np.vstack([Y, box_points])
    return Voronoi(extended_Y), (min_xy, max_xy)


def reassign_to_empty_cells(X_mapped, Y):
    tree = cKDTree(Y)
    assigned_indices = tree.query(X_mapped)[1]
    cell_to_points = defaultdict(list)
    for idx, cell in enumerate(assigned_indices):
        cell_to_points[cell].append(idx)

    occupied = set(cell_to_points.keys())
    all_cells = set(range(len(Y)))
    available_cells = list(all_cells - occupied)

    final_assignments = assigned_indices.copy()
    was_reassigned = np.zeros(len(X_mapped), dtype=bool)  # Track reassignments

    for cell, indices in cell_to_points.items():
        if len(indices) > 1:
            for extra_idx in indices[1:]:  # Keep the first; reassign the rest
                if available_cells:
                    nearest = min(available_cells, key=lambda j: np.linalg.norm(X_mapped[extra_idx] - Y[j]))
                    final_assignments[extra_idx] = nearest
                    was_reassigned[extra_idx] = True
                    occupied.add(nearest)
                    available_cells.remove(nearest)

    return final_assignments, was_reassigned


def compute_hungarian_work(X, Y):
    cost_matrix = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)**2
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()

def visualize_modal_pipeline(X, X_mapped, X_final, Y, final_assignments, was_reassigned, title="Modal Transport Pipeline"):

    from scipy.spatial import Voronoi, voronoi_plot_2d

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Panel 1: Original point sets
    axes[0].scatter(X[:, 0], X[:, 1], c='red', label='X (source)', alpha=0.6)
    axes[0].scatter(Y[:, 0], Y[:, 1], c='blue', label='Y (target)', alpha=0.6)
    axes[0].set_title("1. Original Point Sets")
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True)

    # Panel 2: Moment-matched X̃ with original colors
    axes[1].scatter(X[:, 0], X[:, 1], c='red', alpha=0.6, label='X (source)')
    axes[1].scatter(X_mapped[:, 0], X_mapped[:, 1], c='blue', alpha=0.6, label='Mapped $\~X$')
    for i in range(len(X)):
        axes[1].plot([X[i, 0], X_mapped[i, 0]], [X[i, 1], X_mapped[i, 1]], 'r--', linewidth=0.5)
    axes[1].set_title("2. Moment Matching (Same Color as Panel 1)")

    # Panel 3: Clipped Voronoi Diagram
    vor, (min_xy, max_xy) = compute_clipped_voronoi(Y, np.vstack([X, Y]))
    voronoi_plot_2d(vor, ax=axes[2], show_vertices=False, line_colors='gray', line_width=0.6, line_alpha=0.6, point_size=2)

    # Masking to zoom in
    axes[2].set_xlim(min_xy[0], max_xy[0])
    axes[2].set_ylim(min_xy[1], max_xy[1])

    axes[2].scatter(X_mapped[:, 0], X_mapped[:, 1], c='red', label='Mapped $\~X$', alpha=0.6)
    axes[2].scatter(Y[:, 0], Y[:, 1], c='blue', label='Generators Y', alpha=0.6, edgecolors='k')
    axes[2].set_title("3. Clipped Voronoi in Convex Hull Region")
    axes[2].legend()


   # Panel 4: Fictitious Reassignment
    axes[3].scatter(X_mapped[:, 0], X_mapped[:, 1], c='orange', label='Moment-Matched $\~X$', alpha=0.6)
    
    num_reassigned = 0
    for i in range(len(X)):
        if was_reassigned[i]:
            generator_loc = Y[final_assignments[i]]
            axes[3].scatter(generator_loc[0], generator_loc[1], c='gray', alpha=0.5, s=25)
            axes[3].plot(
                [X_mapped[i, 0], generator_loc[0]],
                [X_mapped[i, 1], generator_loc[1]],
                'k:', linewidth=0.5
            )
            num_reassigned += 1

    axes[3].set_title(f"4. Fictitious Reassignment ({num_reassigned} reassigned)")
    axes[3].legend()
    axes[3].axis('equal')
    axes[3].grid(True)



    # Panel 5: Final snapping
    axes[4].scatter(X_mapped[:, 0], X_mapped[:, 1], c='orange', label='Moment-Matched $\~X$', alpha=0.6)
    axes[4].scatter(Y[:, 0], Y[:, 1], c='blue', label='Generators Y', alpha=0.7, edgecolors='k')
    for i in range(len(X)):
        axes[4].arrow(X_mapped[i, 0], X_mapped[i, 1],
                      X_final[i, 0] - X_mapped[i, 0], X_final[i, 1] - X_mapped[i, 1],
                      head_width=0.2, head_length=0.3, fc='green', ec='green', alpha=0.7, linewidth=0.8)
    axes[4].set_title("5. Final Snapping to Y")
    axes[4].legend()
    axes[4].axis('equal')
    axes[4].grid(True)

    # Hide unused panel
    axes[5].axis('off')

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def modal_transport_analysis(visualize_results=False):
    X, Y = generate_data()
    X_mapped = moment_match(X, Y)
    work_moment_matching = np.sum(np.linalg.norm(X - X_mapped, axis=1)**2)

    _ = compute_clipped_voronoi(Y, np.vstack([X, Y]))

    final_assignments, was_reassigned = reassign_to_empty_cells(X_mapped, Y)
    X_final = Y[final_assignments]
    work_snapping = np.sum(np.linalg.norm(X_mapped - X_final, axis=1)**2)
    work_hungarian = compute_hungarian_work(X, Y)
    total_work_modal = work_moment_matching + work_snapping

    print("----- Work Summary (Corrected) -----")
    print(f"Step 2 - Moment Matching (real):         {work_moment_matching:.4f}")
    print(f"Step 4 - Voronoi Reassignment (fict.):   (excluded from work)")
    print(f"Step 5 - Snap to Generators (real):      {work_snapping:.4f}")
    print(f"Total Work (Modal Method):               {total_work_modal:.4f}")
    print(f"Total Work (Hungarian Method):           {work_hungarian:.4f}")
    print(f"Modal / Hungarian Ratio:                 {total_work_modal / work_hungarian:.4f}")

    if visualize_results:
        visualize_modal_pipeline(X, X_mapped, X_final, Y, final_assignments, was_reassigned)


# Run it
modal_transport_analysis(visualize_results=True)
