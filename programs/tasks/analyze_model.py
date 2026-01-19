from programs.utils.logger_setup import init_shared_logger, set_logger_level
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.arguments import process_args
from programs.utils.common import os_path, create_folder, read_from_json, pretty_dict
from programs.viz.model_plots import load_scores, plot_metrics


def main(args):
    if args.debug_on:
        set_logger_level(10)
        
    expe_folder = args.input_folder
    max_epoch = args.max_epoch

    study_info_path = os_path.join(expe_folder, "study_info")
    first_trial_path = os_path.join(expe_folder, "trial_0")

    if os_path.exists(study_info_path):
        # gets all trial folders
        #trial_folder_names = sorted([f for f in listdir(expe_folder) if f.startswith('trial_')], key=numeric_sort_key)
        best_trial_info = read_from_json(f"{study_info_path}/best_trial_results.json")
        best_trial_folder = f"trial_{best_trial_info['trial']}"
        best_trial_path = os_path.join(expe_folder, best_trial_folder)
        results_path = best_trial_path
    elif os_path.isdir(first_trial_path):
        results_path = first_trial_path
    else:
        results_path = expe_folder


    model_params_path =  os_path.join(results_path, "parameters", "model_params.json")
    if not os_path.exists(model_params_path):
        logger.error(f"Cannot read model params file from: {model_params_path}")
        exit(-1)

    model_params_info = read_from_json(model_params_path)
    model_key = model_params_info.get('model_key')
    solver_key = model_params_info.get('solver_key')
    title_prefix = f"Dataset: {solver_key.upper()} | Model: {model_key.upper()}"

    scores_data = load_scores(results_path, max_epoch=max_epoch)

    output_path = os_path.join(results_path, 'analysis_plots')
    if not os_path.exists(output_path):
        create_folder(output_path)

    logger.info(pretty_dict(model_params_info, label="MODEL_PARAMS"))

    for key in ["training", "validation", "testing"]:
        scores_key = f"{key}_results"
        if not scores_key in scores_data.keys():
            logger.warning(f"Scores data '{scores_key}' is missing or possibly not recorded), skipping...")
            continue
        scores = {f"{key}_results": scores_data[scores_key]}
        new_prefix = rf"Phase: {key.upper()} | {title_prefix}"
        new_path = create_folder(os_path.join(output_path, key))
        plot_metrics(scores, new_path, new_prefix)


if __name__ == "__main__":
    try:
        args = process_args(__file__)
        main(args)
    except Exception as e:
        logger.error(e)

