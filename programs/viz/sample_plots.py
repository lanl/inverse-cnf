from programs.utils.logger_setup import get_logger
from programs.utils.common import defaultdict, plt, np, sns, os_path, mcolor, mtick, as_list, create_file_path
from programs.core.sample_evaluation import ImageContinuousMetrics, ImageSimilarityComparison

prefixes = ["id", "x0", "y0", "x1", "y1"]

label_cmaps = {
    "electrostatic": {
        "charge": "coolwarm",
        "permittivity": "plasma",
        "potential": "turbo"  
    },

    "heat_diffusion": {
        "diffusion": "plasma",
        "initial": "turbo",
        "final": "turbo"  
    }
}

gradient_colors = [
    "#5a002f",  # Deep Magenta
    "#e02070",  # Bright Pink
    "#b13dd1",  # Intense Violet
    "#8855f5",  # Soft Purple
    "#5a40e8",  # Deep Purple 
    "#2d58cc",  # Indigo
    "#008cff",  # Pure Blue
    "#00b6e5",  # Bright Cyan 
    "#00d07a",  # Vibrant Teal
    "#5ac032",  # Lush Green 
    "#0c661c"   # Deep Green
]

gradient_cmap = mcolor.LinearSegmentedColormap.from_list("gradient_cmap", gradient_colors, N=256)

def get_label_cmap(label):
    for cmap_dict in label_cmaps.values():
        for lab, cmap in cmap_dict.items():
            if label.startswith(lab):
                return cmap
    return gradient_cmap

def get_sample_metadata(sample):
    prefixes = ["id", "x0", "y0", "x1", "y1"]   

    def get_sample_keys(prefix):
        sample_keys = [k for k in sample if k.startswith(prefix)] 
        if len(sample_keys) == 1:
            return sample_keys[0]
        elif len(sample_keys) > 1:
            return sorted(sample_keys)
        return None

    sample_keys = {}
    for pre in prefixes:
        keys = get_sample_keys(pre)
        if keys is None:
            continue
        sample_keys[pre] = keys

    channel_labels = {
        "id": sample_keys["id"].split("_", 1)[1],
        "x": [k.split("_", 1)[1] 
                for k in sample_keys["x0"]] 
                    if isinstance(sample_keys["x0"], list) 
                    else [sample_keys["x0"].split("_", 1)[1]],
        "y": sample_keys["y0"].split("_", 1)[1]
    }

    if get_logger().logger.level == 10:
        log_lines = [f"SHAPES FOR SAMPLE ID: {sample[sample_keys['id']]}\n\n"]
        for val in sample_keys.values():
            for k in as_list(val):
                typ = type(sample[k])
                shape = sample[k].shape if typ == np.ndarray else 1
                log_lines.append(f"{k}: shape = {shape}, type = {typ}")
        get_logger().debug("\n".join(log_lines) + "\n")

    return sample_keys, channel_labels


def plot_generated_images(sample_dicts: list, epoch: int, plot_path: str):
    """
    Plot input, generated, and solved images from structured sample_dicts.
    Each sample dict contains:
        - id_<...>
        - x0_<channel name>: (H,W)
        - x1_<channel name>: (H,W) or (S,H,W)
        - y0_<channel name>: (H,W)
        - y1_<channel name>: (H,W) or (S,H,W)

    S: samples, H: height, W: width
    """
    first = sample_dicts[0]
    keys, labels = get_sample_metadata(first)

    id_label = labels["id"]

    col_slices = []
    col_cmaps = []
    col_labels = as_list(labels["x"]) + as_list(labels['y'])
    col_keys = as_list(keys["x0"]) + as_list(keys["y0"])

    for key, lab in zip(col_keys, col_labels):
        arr = first[key]
        if np.ndim(arr) == 3:
            for i in range(arr.shape[0]):
                col_slices.append((key, i))
                col_cmaps.append(get_label_cmap(lab))
        elif np.ndim(arr) == 1:
            for i in range(arr.shape[0]):
                col_slices.append((key, i))
                col_cmaps.append(get_label_cmap(lab))
        else:
            col_slices.append((key, None))
            col_cmaps.append(get_label_cmap(lab))

    num_cols = len(col_slices)
    bbox_props = dict(boxstyle="round,pad=1.0", edgecolor="black", facecolor="none", linewidth=2)

    for sample in sample_dicts:
        uid = sample[keys["id"]]
        y0 = sample[keys["y0"]]

        # row 0: extract x0 channels
        x0_images = []
        for k, c in col_slices:
            img = sample[k]
            if c is not None:
                img = img[c]
            x0_images.append(img)
        x0_images.append(y0)

        # x1 (generated inputs)
        S = min(5, len(sample[keys["x1"][0]]))  # Limit number of samples to plot
        x1_rows = []

        for s in range(S):  # Iterate through each sample (s-th sample)
            row = []
            for k in keys["x1"]:
                arr = sample[k]  # Can be scalar, shape (S, C, H, W), or (S, H, W)
                if isinstance(arr, np.ndarray):  # Image array
                    if arr.ndim == 3:  # (S, H, W)
                        row.append(arr[s])
                    elif arr.ndim == 1:  # (S,)
                        row.append(arr[s])
                    else:
                        raise ValueError(f"Unexpected shape for x1[{k}]: {arr.shape}")
                elif isinstance(arr, (int, float)):
                    row.append(arr)
                else:
                    raise TypeError(f"Expected ndarray or scalar for x1[{k}], got {type(arr)}")
            
            x1_rows.append(row)

        if "y1" in keys:
            # y1 (solved targets)
            y1_arr = sample[keys["y1"]]
            if y1_arr.ndim == 3:
                y1_images = [y1_arr[i] for i in range(y1_arr.shape[0])]
            else:
                y1_images = [y1_arr]
            # Add y1 images to each row
            for i in range(S):
                x1_rows[i].append(y1_images[i])

        fig, axes = plt.subplots(S + 1, num_cols, figsize=(5.5 * num_cols, 5 * (S + 1)))

        n_cols = len(col_labels)
        used_axes = set()

        # row 0 (x0 and y0)
        for j, (img, label, cmap) in enumerate(zip(x0_images, col_labels, col_cmaps)):
            ax = axes[0, j]
            if np.ndim(img) == 0:
                ax.text(0.5, 0.5, f"{label}: {img}", ha='center', va='center', fontsize=16, color="black", bbox=bbox_props)
            else:
                im = ax.imshow(img, cmap=cmap)

                if img.shape[-1] > 6:
                    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3g}"))
                else:
                    for x in range(img.shape[-2]):
                        for y in range(img.shape[-1]):
                            ax.text(y, x, f'{img[x, y]:.4f}', ha='center', va='center', color='white', fontsize=14)

            prefix = f"X{j+1}" if j < (n_cols - 1) else "Y"
            ax.set_title(f"Original {prefix}: {label}", fontsize=18)
            ax.axis('off')
            used_axes.add(ax)

        # rows 1..S (x1 and y1)
        for i, row_imgs in enumerate(x1_rows):
            for j, (img, label, cmap) in enumerate(zip(row_imgs, col_labels, col_cmaps)):
                ax = axes[i + 1, j]
                if np.ndim(img) == 0:
                    ax.text(0.5, 0.5, f"{label}: {img}", ha='center', va='center', fontsize=16, color="black", bbox=bbox_props)
                else:
                    im = ax.imshow(img, cmap=cmap)
                    if img.shape[-1] > 6:
                        cbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
                        cbar.ax.tick_params(labelsize=16)
                        cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3g}"))
                    else:
                        for x in range(img.shape[-2]):
                            for y in range(img.shape[-1]):
                                ax.text(y, x, f'{img[x, y]:.4f}', ha='center', va='center', color='white', fontsize=14)

                prefix = "Generated" if "y1" not in keys or j < (n_cols - 1) else "Solved"
                ax.set_title(f"{prefix} #{i + 1}: {label}", fontsize=18)
                ax.axis('off')
                used_axes.add(ax)

        for ax in axes.flatten():
            if ax not in used_axes:
                fig.delaxes(ax)

        plt.suptitle(f"Testing Set Generation - Epoch #{epoch} - {id_label.title()} #{uid}", fontsize=22, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_file = create_file_path(os_path.join(plot_path, f"{id_label}_{uid}"), 
                                    f"generated_images_epoch_{epoch}_{id_label}_{uid}.png")
        plt.savefig(plot_file)
        plt.close()

        get_logger().info(f"Saved (generated X, solved Y) plot for {id_label} #{uid} to: {plot_file}")


def plot_reconstructed_images(sample_dicts: list, epoch: int, phase:str, plot_path: str):
    """
    Plot input, reconstructed, and projected images from structured sample_dicts.
    Each sample dict contains:
        - id_<...>
        - x0_<channel name>: (H,W)
        - x1_<channel name>: (H,W)
        - y0_<channel name>: (H,W)
        - y1_<channel name>: (H,W)

    H: height, W: width

    If a channel is scalar (0-D or 1-D), show its value instead.
    """
    first = sample_dicts[0]
    keys, labels = get_sample_metadata(first)

    col_slices = []
    col_cmaps = []
    col_labels = as_list(labels["x"]) + as_list(labels['y'])
    
    xy0_keys = as_list(keys["x0"]) + as_list(keys["y0"])
    xy1_keys = as_list(keys["x1"]) + as_list(keys["y1"])

    for key, lab in zip(xy0_keys, col_labels):
        arr = first[key]
        if isinstance(arr, (int, float)) or (isinstance(arr, np.ndarray) and arr.ndim == 2):
            # shape (H, W) or scalar only
            col_slices.append((key, None))
            col_cmaps.append(get_label_cmap(lab))
        else:
            raise ValueError(f"Unexpected shape for {key}: {arr.shape}")

    num_cols = len(col_slices)
    bbox_props = dict(boxstyle="round,pad=1.0", edgecolor="black", facecolor="none", linewidth=2)

    for sample in sample_dicts:
        uid = sample[keys["id"]]

        # row 0: x0 + y0 (original)
        xy0_images = []
        for k, c in col_slices:
            img = sample[k]
            if c is not None:
                img = img[c]
            xy0_images.append(img)

        # row 1: x1 + y1 (reconstructed)
        xy1_images = []
        for k in xy1_keys:
            arr = sample[k]  # shape (H, W)
            if isinstance(arr, (int, float)) or (isinstance(arr, np.ndarray) and arr.ndim == 2): 
                xy1_images.append(arr)
            else:
                raise ValueError(f"Unexpected shape for {k}: {arr.shape}")

        fig, axes = plt.subplots(3, num_cols, figsize=(5.5 * num_cols, 15))

        # row 0 (x0 and y0)
        for j, (img, label, cmap) in enumerate(zip(xy0_images, col_labels, col_cmaps)):
            ax = axes[0, j]
            if isinstance(img, (int, float)):
                ax.text(0.5, 0.5, f"{label} = {img:.4f}", ha="center", va="center", fontsize=16, bbox=bbox_props)   
            else:
                im = ax.imshow(img, cmap=cmap)

                if img.shape[-1] > 6:
                    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3g}"))
                else:
                    for x in range(img.shape[-2]):
                        for y in range(img.shape[-1]):
                            ax.text(y, x, f'{img[x, y]:.4f}', ha='center', va='center', color='white', fontsize=14)
            prefix = f"X{j}" if j < (num_cols - 1) else "Y"
            ax.set_title(f"Original {prefix}: {label}", fontsize=18)
            ax.axis('off')

        # row 1 (x1 and y1)
        for j, (img, label, cmap) in enumerate(zip(xy1_images, col_labels, col_cmaps)):
            ax = axes[1, j]
            if isinstance(img, (int, float)):
                ax.text(0.5, 0.5, f"{label} = {img:.4f}", ha="center", va="center", fontsize=16, bbox=bbox_props)
            else:
                im = ax.imshow(img, cmap=cmap)

                if img.shape[-1] > 6:
                    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3g}"))
                else:
                    for x in range(img.shape[-2]):
                        for y in range(img.shape[-1]):
                            ax.text(y, x, f'{img[x, y]:.4f}', ha='center', va='center', color='white', fontsize=14)
            prefix = f"Reconstructed X{j}" if j < (num_cols - 1) else "Projected Y"
            ax.set_title(f"{prefix}: {label}", fontsize=18)
            ax.axis('off')

        for j in range(num_cols):
            img0 = xy0_images[j]
            img1 = xy1_images[j]
            label = col_labels[j]
            ax = axes[2, j]

            if isinstance(img0, (int, float)) or isinstance(img1, (int, float)):
                # For scalars, show difference
                diff_val = float(img1) - float(img0)
                ax.text(0.5, 0.5, f"Δ = {diff_val:.2g}", ha="center", va="center", fontsize=16, bbox=bbox_props)
                label_title = f"Signed Diff: {label}"

            else:
                # Compare x1 vs y1 using SSIM
                diff_img = ImageSimilarityComparison._signed_difference(img0, img1)
                im = ax.imshow(diff_img, cmap=gradient_cmap)
                label_title = f"Signed Diff: {label}"

                if diff_img.shape[-1] > 6:
                    cbar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
                    cbar.ax.tick_params(labelsize=16)   
                    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3g}"))
                else:
                    for i in range(img.shape[-2]):
                        for j in range(img.shape[-1]):
                            ax.text(j, i, f'{diff_img[i, j]:.2g}', ha='center', va='center', color='white', fontsize=14)
            
            ax.set_title(label_title, fontsize=18)
            ax.axis('off')

        plt.suptitle(f"{phase.title()} Set Reconstruction - Epoch {epoch} – {labels['id'].title()} #{uid}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plot_file = create_file_path(plot_path, f"reconstructed_images_epoch_{epoch}_{labels['id']}_{uid}.png")
        plt.savefig(plot_file)
        plt.close()
        get_logger().debug(f"Saved (reconstructed X, projected Y) plot for {labels['id']} #{uid} to: {plot_file}")


def plot_latent_images(z_stack: list, epoch: int, phase:str, plot_path: str):
    """
    Plot latent images
    If a channel is scalar (0-D or 1-D), show its value instead.
    """
    bbox_props = dict(boxstyle="round,pad=1.0", edgecolor="black", facecolor="none", linewidth=2)

    for z_record in z_stack:
        id = z_record['id']
        z_chan = z_record['z']
        C, H, W = z_chan.shape
        fig, axes = plt.subplots(1, C, figsize=(C*6, 6))

        for i in range(C):
            ax = axes[i]
            z_img = z_chan[i]

            if isinstance(z_img, (int, float)):
                ax.text(0.5, 0.5, f"Z Latent = {z_img:.4g}", ha="center", va="center", fontsize=16, bbox=bbox_props)
            else:
                im = ax.imshow(z_img, cmap=gradient_cmap)

                if z_img.shape[-1] > 6:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                    cbar.ax.tick_params(labelsize=16)
                else:
                    for x in range(z_img.shape[-2]):
                        for y in range(z_img.shape[-1]):
                            ax.text(y, x, f'{z_img[x, y]:.4g}', ha='center', va='center', color='white', fontsize=14)

            ax.set_title(f'Channel {i+1}')
            ax.axis('off')

        plt.suptitle(f"{phase.title()} Set Latent Z - Epoch {epoch} - random_seed #{id}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plot_file = create_file_path(plot_path, f"latent_images_epoch_{epoch}_random_seed_{id}.png")
        plt.savefig(plot_file)
        plt.close()

        get_logger().debug(f"Saved latent images plot for random_seed #{id} to: {plot_file}")


def get_data_ranges(sample_dicts: list, keys: dict[str, list|str], labels:dict[str, list|str]):

    global_data_extrema = defaultdict(lambda: (float('inf'), float('-inf')))


    xy0_keys = as_list(keys['x0']) + as_list(keys['y0'])
    xy1_keys = as_list(keys['x0']) + as_list(keys['y0'])
    xy_labels = as_list(labels['x']) + as_list(labels['y'])

    for sample in sample_dicts:
        for i, (k0, k1, lab) in enumerate(zip(xy0_keys, xy1_keys, xy_labels)):
            img0 = sample[k0]
            img1 = sample[k1]
            current_min, current_max = global_data_extrema[lab]
            current_min = min(current_min, img0.min(), img1.min())
            current_max = max(current_max, img0.max(), img1.max())
            global_data_extrema[lab] = (current_min, current_max)

    global_data_ranges = {key:max(val)-min(val) for key, val in global_data_extrema.items()}

    return global_data_ranges


def compute_sample_metrics(sample_dicts: list, keys: dict[str, list | str], labels: dict[str, list | str], data_ranges: dict[str, float]|None=None):
    """
    For each sample in sample_dicts, compute scores between:
        - x0_<input> and each x1_<input>[s]
        - y0_<target> and each y1_<target>[s]
    """

    if data_ranges is None:
        data_ranges = get_data_ranges(sample_dicts, keys, labels)

    scores_by_sample = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def compute_scores_for_pair(a, b, sample_id, image_name, metric_dict, data_range=None):
        scores = ImageContinuousMetrics.all_metrics(a, b, data_range)
        for metric, score in scores.items():
            metric_dict[sample_id][metric][image_name].append(score)

    # Merge x0 and y0 keys and x1 and y1 keys
    xy0_keys = as_list(keys['x0']) + as_list(keys['y0'])
    xy1_keys = as_list(keys['x1']) + as_list(keys.get('y1', [])) 
    xy_labels = as_list(labels['x']) + as_list(labels['y'])


    for sample in sample_dicts:
        sample_id = sample[keys["id"]]
        
        # Get S: # samples 
        S = sample[keys['x1'][0]].shape[0]

        # metrics for x0 and x1
        for i, (k0, k1, lab) in enumerate(zip(xy0_keys, xy1_keys, xy_labels)):
            img0 = sample[k0]
            img1 = sample[k1]

            if k0.startswith("x0_"):
                img_name = k0.replace("x0_", f"x{i+1}_")
            if k0.startswith("y0_"):
                img_name = k0.replace("y0_", f"y_")

            for j in range(S):
                compute_scores_for_pair(img0, img1[j], sample_id, img_name, scores_by_sample, data_range = data_ranges[lab])

    return scores_by_sample


def plot_generated_sample_scores(sample_dicts, epoch, out_dir, data_ranges=None):
    keys, labels = get_sample_metadata(sample_dicts[0])

    scores_by_sample_metric_image = compute_sample_metrics(sample_dicts, keys, labels, data_ranges=data_ranges)
    id_label = labels["id"]

    num_samples = len(scores_by_sample_metric_image)
    
    for sample_id, scores_by_metric_image in scores_by_sample_metric_image.items():

        for metric_name, scores_by_image in scores_by_metric_image.items():

            metric_label = metric_name.upper()

            num_images = len(scores_by_image)
            rec_lb, rec_ub = ImageContinuousMetrics.get_metric_bounds(metric=metric_name)
            fig, axes = plt.subplots(num_images, 1, figsize=(16, 8 * num_images))
            colors = sns.color_palette("husl", n_colors=num_images)

            for i, (image_name, image_scores) in enumerate(scores_by_image.items()):

                scores = [score for score in image_scores if not np.isnan(score)] 
                num_total_scores = len(scores)
                num_unique_scores = len(set(scores))
                label_parts = image_name.title().split("_",1)
                prefix = "Generated" if "y1" not in keys or i < num_images-1 else "Solved"
                image_label = f"Original vs {prefix} {label_parts[0]}: {label_parts[1]}" 

                if num_unique_scores == 1:
                    plot_msg = (f"Zero variance between {metric_label} values" 
                                if np.isfinite(scores[0]) 
                                else f"Cannot compute {metric_label} for scalar values")
                    axes[i].set_title(image_label, fontsize=18)
                    axes[i].text(0.5, 0.5, plot_msg, ha="center", va="center", fontsize=16)
                    axes[i].grid(False)
                    continue
                
                if num_total_scores == 0:
                    axes[i].set_title(image_label, fontsize=18)
                    axes[i].text(0.5, 0.5, f"No valid {metric_label} values", ha="center", va="center", fontsize=16)
                    axes[i].grid(False)
                    continue

                sns.kdeplot(scores, ax=axes[i], fill=True, color=colors[i], alpha=0.5, warn_singular=False)
                axes[i].set_title(image_label, fontsize=20)
                axes[i].set_xlabel(metric_label, fontsize=18)
                axes[i].set_ylabel("Density", fontsize=18)
                axes[i].grid(True)

                curr_lb, curr_ub = axes[i].get_xlim()
                if rec_lb is not None and rec_lb < curr_lb:
                    curr_lb = rec_lb
                if rec_ub is not None and rec_ub > curr_ub:
                    curr_ub = rec_ub
                margin = (curr_ub - curr_lb) * 0.05
                axes[i].set_xlim(left=curr_lb-margin, right=curr_ub+margin)


            plt.suptitle(f"Original vs Generated (N={num_total_scores}) {metric_label}: Epoch #{epoch} - {id_label.title()} #{sample_id}", fontsize=24)
            plot_file = create_file_path(os_path.join(out_dir, f"{id_label}_{sample_id}"), f"kde_plot_epoch_{epoch}_{id_label}_{sample_id}_{metric_name.lower()}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(plot_file)
            plt.close()

            get_logger().info(f"Saved ({metric_label} KDE) plot for {id_label} #{sample_id} to: {plot_file}")