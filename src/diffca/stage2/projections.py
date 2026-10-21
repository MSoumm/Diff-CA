"""2-D PCA and UMAP projections of the salient codes h (and of the mean common token), as in the paper's App. G.

Files: <output_dir>/{pca,umap}/<epoch>/{pca,umap}_{cls,context,cls_positive,cls_fit_sub12}_epoch_<epoch+1>.{png,pdf}
"cls_fit_sub12": the projection is fitted on sublabels 1 and 2 only, then applied to every sample.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from umap import UMAP

FONT_SCALE = 3.0
SUBLABEL_NAMES = {0: "No Glasses", 1: "Reading glasses", 2: "Sunglasses"}


def _legend(scatter, values, title, names=None):
    handles, _ = scatter.legend_elements()
    unique = np.unique(values)
    if len(handles) > 0 and len(unique) > 1:
        labels = [names.get(v, str(v)) for v in unique] if names is not None else [str(v) for v in unique]
        plt.legend(handles=handles, title=title, labels=labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=9 * FONT_SCALE, title_fontsize=10 * FONT_SCALE)


def _scatter(embedding, colors, path, title, method, cmap="viridis", legend=None, continuous=False):
    """One square scatter plot, saved as .png and .pdf. `legend`: (title, names) or None."""
    plt.figure(figsize=(7, 7))
    if continuous:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="viridis", s=10, alpha=0.6,
                              vmin=float(np.min(colors)), vmax=float(np.max(colors)))
        cbar = plt.colorbar(scatter)
        cbar.set_label("Yaw", fontsize=10 * FONT_SCALE)
        cbar.ax.tick_params(labelsize=9 * FONT_SCALE)
    else:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, s=10, alpha=0.6)
        if legend is not None:
            _legend(scatter, colors, *legend)
    plt.title(title, fontsize=12 * FONT_SCALE)
    plt.xlabel(f"{method} Dimension 1", fontsize=10 * FONT_SCALE)
    plt.ylabel(f"{method} Dimension 2", fontsize=10 * FONT_SCALE)
    plt.tick_params(axis="both", labelsize=9 * FONT_SCALE)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved {path}")


def _umap(seed, n_neighbors=30):
    return UMAP(n_components=2, random_state=seed, metric="euclidean", n_neighbors=n_neighbors, min_dist=0.0)


def generate_projections(method, all_labels, all_sublabels, all_cls, all_context_tokens, epoch, output_dir, seed):
    """`method`: "pca" or "umap". Inputs are lists of per-batch tensors (labels, sublabels, h, mean common token)."""
    if not output_dir:
        return
    is_umap = method == "umap"
    name = "UMAP" if is_umap else "PCA"
    print(f"--- Generating {name} visualizations ---")

    labels = torch.cat(all_labels).numpy()
    sublabels = torch.cat(all_sublabels).numpy()
    cls = torch.cat(all_cls).numpy()
    context = torch.cat(all_context_tokens).numpy()
    out_dir = os.path.join(output_dir, method, str(epoch))
    os.makedirs(out_dir, exist_ok=True)

    def path(tag):
        return os.path.join(out_dir, f"{method}_{tag}_epoch_{epoch + 1}.png")

    label_legend = ("Labels", None) if is_umap else None
    sublabel_legend = ("Sub-Labels", SUBLABEL_NAMES) if is_umap else None
    float_sublabels = np.issubdtype(sublabels.dtype, np.floating)

    def fit_transform(data, tag, n_neighbors=30):
        if is_umap:
            return _umap(seed, n_neighbors).fit_transform(data)
        pca = PCA(n_components=2, random_state=seed)
        embedding = pca.fit_transform(data)
        if tag is not None:
            print(f"PCA Explained Variance Ratios for CLS Tokens{tag}: {pca.explained_variance_ratio_}")
        return embedding

    try:
        _scatter(fit_transform(cls, ""), labels, path("cls"), "UMAP of Salient Vectors" if is_umap else "PCA of Salient Vectors",
                 name, legend=label_legend)
    except Exception as e:
        print(f"Could not generate {name} for CLS tokens: {e}")

    try:
        _scatter(fit_transform(context, None), labels, path("context"), f"{name} of Mean Context Tokens - Epoch {epoch + 1}",
                 name, legend=label_legend)
    except Exception as e:
        print(f"Could not generate {name} for context tokens: {e}")

    positive = labels == 1
    if np.any(positive):
        try:
            cls_pos, sub_pos = cls[positive], sublabels[positive]
            embedding = fit_transform(cls_pos, " (Positive Class)", n_neighbors=min(30, len(cls_pos) - 1))
            _scatter(embedding, sub_pos, path("cls_positive"), f"{name} of Salient Vectors", name, cmap="jet",
                     legend=sublabel_legend, continuous=np.issubdtype(sub_pos.dtype, np.floating))
        except Exception as e:
            print(f"Could not generate {name} for positive CLS tokens: {e}")

    if not float_sublabels:
        fit_mask = np.isin(sublabels, [1, 2])
        if np.any(fit_mask):
            try:
                if is_umap:
                    reducer = _umap(seed).fit(cls[fit_mask])
                else:
                    reducer = PCA(n_components=2, random_state=seed).fit(cls[fit_mask])
                    print(f"PCA Explained Variance Ratios for CLS Tokens (fit sublabels 1&2): {reducer.explained_variance_ratio_}")
                _scatter(reducer.transform(cls), sublabels, path("cls_fit_sub12"), f"{name} of Salient Vectors", name,
                         cmap="jet", legend=sublabel_legend)
            except Exception as e:
                print(f"Could not generate {name} fit on sublabels 1&2: {e}")
