from programs.utils.logger_setup import get_logger
from programs.utils.common import np, sns, plt, os_path, listdir, extract_filename_number, read_from_json, numeric_sort_key


def format_metric_name(metric):
    is_loss = "loss" in metric.lower()
    title_postfix = metric if is_loss else f"Metric: {metric.upper()}"
    metric_label = metric.split(': ', 1)[1] if is_loss else metric.upper()
    return fr"{title_postfix}", fr"{metric_label}"

def format_file_name(title):
    return title.replace(' ', '').replace(':', '-').replace('—', '_').replace('|', '_').replace('\n', '_').lower()


def load_scores(folder, max_epoch:int|None=None):
    scores_data = {}

    for phase in ['training_results', 'validation_results', 'testing_results']:
        phase_scores = []
        folder_path = os_path.join(folder, phase, 'model_scores')
        
        if not os_path.exists(folder_path):
            get_logger().warning(f"Could not find model_scores folder in: {folder_path}")
            continue

        score_files = sorted([f for f in listdir(folder_path) if f.endswith('.json')], key=numeric_sort_key)

        if not score_files:
            get_logger().warning(f"No model score files exist in directory: {folder_path}")
            continue
        
        for file_name in score_files:
            epoch = extract_filename_number(file_name)
            if epoch and max_epoch and epoch > max_epoch:
                break
            file_path = os_path.join(folder_path, file_name)
            
            try:
                file_scores = read_from_json(file_path)

                overall = [s for s in file_scores if s.get('target') == 'overall']
                if overall:
                    score = overall[0]
                else:
                    score = file_scores[0]

                # for old experiments
                if 'loss_function' in score:
                    loss = score['loss_function']
                    loss_metrics = {
                        f"Loss: AVG_{loss['loss_name']}": loss['avg_loss'],
                        f"Loss: TOTAL_{loss['loss_name']}": loss['total_loss']
                    }
                    score['metrics'] = {
                        **loss_metrics,
                        **score.get('metrics', {})
                    }

                # for new experiments
                elif 'loss' in score:
                    loss_scores = score['loss']
                    loss_name = loss_scores['function']['name']
                    loss_summary = loss_scores['summary']

                    loss_metrics = {
                        f"Loss: AVG_{loss_name}": loss_summary['avg_loss'],
                        f"Loss: TOTAL_{loss_name}": loss_summary['total_loss']
                    }
                    

                    score['metrics'] = {
                        **loss_metrics,
                        **score.get('metrics', {})
                    }

                    if phase == 'training_results':
                        y_loss_scores = loss_scores['partial']['y_loss']
                        beta_weight = y_loss_scores['weighted']['beta_weight']
                        beta_term = y_loss_scores['weighted']['beta_term']
                        beta_weight = {
                            fr'{beta_term} Weight': beta_weight,
                        }
                        score['extra'] = beta_weight

                phase_scores.append(score)

            except Exception as e:
                get_logger().warning(f"Error reading file {file_path}: {e}")

        if phase_scores:
            scores_data[phase] = sorted(phase_scores, key=lambda x: x['epoch'])
        else:
            get_logger().warning(f"No scores saved for phase: {phase}")
    
    return scores_data


def plot_metrics(scores_data, output_folder, title_prefix):
    if not scores_data:
        get_logger().warning("No valid score data to plot")
        return

    first_phase = next(iter(scores_data.values()), [])
    if not first_phase:
        return

    first_entry = first_phase[0]
    metrics = list(first_entry['metrics'].keys())
    max_epoch = max([entry['epoch'] for phase_scores in scores_data.values() for entry in phase_scores])

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

    phase_styles = {
        "Training":   {"color": "forestgreen", "linestyle": "-",  "linewidth": 4, "zorder": 1},
        "Validation": {"color": "darkorange","linestyle": "--", "linewidth": 4, "zorder": 2},
        "Testing":    {"color": "royalblue", "linestyle": ":",  "linewidth": 5, "zorder": 3}
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(20, 12))
        ax2 = None
        is_loss_metric = 'loss' in metric.lower()
        min_val_loss = 0
        min_val_epoch = 0

        for phase, scores in scores_data.items():
            epochs = [entry['epoch'] for entry in scores]
            metric_values = [entry['metrics'][metric] for entry in scores]

            phase_name = phase.split("_")[0].capitalize()
            phase_style = phase_styles.get(phase_name, {})

            sns.lineplot(
                x=epochs, 
                y=metric_values, 
                label=f"{phase_name}: {'Loss' if is_loss_metric else 'Metric'} Value",
                color=phase_style["color"],
                markers=False,
                linestyle=phase_style["linestyle"],
                linewidth=phase_style["linewidth"],
                zorder=phase_style["zorder"], 
                ax=ax
            )
            
            if not is_loss_metric: 
                continue
            
            # extra art for loss metrics
            if phase_name == "Training" and "extra" in scores[0]:
                beta_key = next(iter(scores[0]["extra"]))
                beta_values = [entry["extra"][beta_key] for entry in scores]

                if ax2 is None:
                    ax2 = ax.twinx()
                    ax2.set_ylabel(beta_key, labelpad=10, fontweight='bold')

                    sns.lineplot(
                        x=epochs, 
                        y=beta_values, 
                        label=f"{phase_name}: {beta_key}",
                        color="darkorchid",
                        markers=False,
                        linestyle=":",
                        linewidth=phase_style["linewidth"],
                        zorder=phase_style["zorder"]-1, 
                        ax=ax2
                    )

            elif phase_name == "Validation":
                min_idx = metric_values.index(min(metric_values))
                min_val_loss = metric_values[min_idx]
                min_val_epoch = epochs[min_idx]

        if is_loss_metric:
            if min_val_epoch > 0:
                color='mediumvioletred'
                ax.scatter(x=min_val_epoch, y=min_val_loss, color=color, marker="*", s=150, label=f'Validation: Minimum Loss', zorder=4)
                ax.annotate(f"Epoch #{min_val_epoch} = {min_val_loss:g}", 
                            (min_val_epoch, min_val_loss), 
                            textcoords="offset points", 
                            xytext=(0, 60), 
                            ha='center',
                            fontsize=18, 
                            color='black', 
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    linestyle='--', 
                                    linewidth=2.5, 
                                    edgecolor=color, 
                                    facecolor=(1, 1, 1, 0.7)),
                            arrowprops=dict(arrowstyle='->', 
                                            linestyle="-", 
                                            linewidth=2.5, 
                                            connectionstyle='angle,angleA=0,angleB=90',
                                            shrinkB=10,
                                        color=color))

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="best")

            elif ax2 is not None:
                y2_limit = ax2.get_ylim()
                y2_min, y2_max = max(0, y2_limit[0]), min(1, y2_limit[1])
                y2_range = y2_max - y2_min
                y2_buffer = y2_range * 0.05
                y2_ticks = np.linspace(y2_min, y2_max, num=11)
                y2_labels = [rf"{y:.2f}" for y in y2_ticks]

                ax2.set_ylim(y2_min - y2_buffer, y2_max + y2_buffer)
                ax2.set_yticks(y2_ticks)
                ax2.set_yticklabels(y2_labels)
    
                ax2.tick_params(axis='y', which="major", labelrotation=-20)

                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h1+h2, l1+l2, loc="best")
                ax.get_legend().remove()


        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        y_buffer = y_range * 0.05
        precision = 2 if y_range > 0.2 else 3
        y_ticks = np.linspace(y_min, y_max, num=11)
        y_labels = [rf"{y:.{precision}f}" for y in y_ticks]

        ax.set_ylim(y_min-y_buffer, y_max+y_buffer)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        num_ticks = min(21, max_epoch)  # Ensure we don't generate more ticks than needed
        x_ticks = np.linspace(0, max_epoch, num=num_ticks, dtype=int)

        x_labels = [rf"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}" for x in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        ax.tick_params(axis='y', which="major", labelrotation=20)


        title_postfix, metric_label = format_metric_name(metric)
        full_title = f"{title_prefix} | {title_postfix}"
        file_prefix = format_file_name(full_title)

        ax.set_title(full_title, pad=18, fontweight='bold')
        ax.set_xlabel("Epoch", labelpad=10, fontweight='bold')
        ax.set_ylabel("Loss Value" if 'loss' in metric.lower() else rf"{metric_label} Value", labelpad=10, fontweight='bold')    

        ax.grid(True)
        fig.tight_layout()

        file_prefix = full_title.replace(' ', '').replace(':', '-').replace('—', '_').replace('|', '_').replace('\n', '_').lower()
        
        png_file_path = os_path.join(output_folder, f"{file_prefix}_plot.png")
        plt.savefig(png_file_path, dpi='figure')
        get_logger().info(f"Saved static plot for {metric} to {png_file_path}")
        plt.close()