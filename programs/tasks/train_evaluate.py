from programs.utils.logger_setup import init_shared_logger, set_logger_level, log_execution_time
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.arguments import process_args
from programs.utils.common import pt, np, pretty_dict
from programs.utils.device_setup import SetupDevice
from programs.utils.experiment_setup import setup_model_instance, get_default_config
from programs.core.data_manager import BatchSubset
from programs.core.optuna_manager import optuna, optimize
from programs.core.experiment_tasks import train_validate_test


@log_execution_time
def main(args):
    if args.debug_on:
        set_logger_level(10)

    # process model defaults and hyperparams
    model_config = get_default_config(args.model_key)

    model_config['device'] = SetupDevice.setup_torch_device(
        args.num_tasks,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )


    if 'conv_params' in model_config and model_config['conv_params']:
        model_config['conv_params']['kernel'] = args.conv_kernel
        model_config['conv_params']['stride'] = args.conv_stride

    if args.hybrid_loss_functions is not None:
        model_config['loss_function'] = "weighted-hybrid"

    hyperparams = {}
    for param in model_config.get('hyperparams', []):
        val = getattr(args, param, None)
        if val is not None:
            hyperparams[param] = val

    model_config.update(hyperparams)

    # get dataloaders for each subset
    subset = BatchSubset(args.input_folder, "training")
    if not subset:
        raise RuntimeError("Cannot load batch subset for 'training'")

    batch_loaders = {}
    for label in subset.subset_names:
        sub = BatchSubset(args.input_folder, label)
        batch_loaders[label] = sub.get_dataloader(
            num_workers=max(args.num_tasks - 1, 1)
        )

    # setup experiment config dict
    expe_config = {
        # epoch frequencies
        "epoch_frequency": {
            "checkpoint": args.checkpoint_frequency or 0,
            "training": 1,
            "validation": args.validation_frequency,
            "testing": args.testing_frequency
        },
        # solver & sampling
        "random_seed": args.random_seed,
        "num_epochs": args.num_epochs,
        "testing_samples": args.testing_samples,
        "generation_samples": args.generation_samples,
        "generation_noise": args.generation_noise,
        "generation_limit": args.generation_limit,
        "solver_key": args.solver_key,
        "plot_debug_images": args.plot_debug_images_on,
        "hybrid_loss_params": {
            "functions": args.hybrid_loss_functions,
            "weights": args.hybrid_loss_weights
        },
        "multi_scale_params": {
            "kernel": args.multi_scale_kernel,
            "weights": args.multi_scale_weights
        },
        "beta_schedule_params": {
            "mode": args.beta_schedule_mode,
            "min_value": args.beta_value_range[0],
            "max_value": args.beta_value_range[1],
            "warmup_value": args.beta_warmup_epochs,
            "warmup_type": "fixed" if args.beta_warmup_epochs >= 1.0 else "fraction",
        },  
        "affine_log_params": {
            "transform": args.affine_log_transform,
            "bounds": args.affine_log_bounds,
        },
        # dataset metadata
        "subset_names": subset.subset_names,
        "subset_sizes": subset.subset_sizes,
        "unique_id_name": subset.unique_id_name,
        "transform_method": subset.transform_method,
        "data_transforms": subset.data_transforms,
        "data_groups": subset.data_groups
    }

    # merge model config with expe config
    expe_config.update(model_config)

    logger.info(pretty_dict(expe_config, label="EXPE_CONFIG"))

    # single training 
    if not args.tuning_on:

        model = setup_model_instance(expe_config, args.output_folder)
        train_validate_test(model, expe_config, batch_loaders, args.output_folder)

    # optuna trials
    else:
        # get number of total trials
        num_trials = np.prod([len(value) for value in hyperparams.values()])
        trial_patience = min(args.trial_patience, num_trials)

        if args.trial_patience > num_trials:
            logger.warning(f"trial_patience ({args.trial_patience}) exceeds maximum possible trials ({num_trials}),defaulting trial_patience to the maximum.")

        # setup study config
        study_config = {
            'objective': f"{model_config['loss_prefix']}_loss",
            'direction': 'minimize',
            'sampler': optuna.samplers.TPESampler(),
            'num_trials': num_trials,
            'early_stop': args.earlystop_on,
            'pruner_option': args.pruner_option,
            'threshold_upper': args.threshold_upper,
            'min_delta': args.min_delta,
            'epoch_patience': args.epoch_patience,
            'trial_patience': trial_patience,
            'hyperparams':hyperparams
        }

        logger.info(pretty_dict(study_config, label="STUDY_CONFIG"))

        optimize(expe_config, study_config, batch_loaders, args.output_folder)


if __name__ == "__main__":
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args(__file__)
        main(args)
    except Exception as e:
        logger.error(e)
