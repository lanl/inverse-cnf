from programs.utils.logger_setup import init_shared_logger, set_logger_level, log_execution_time
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.common import pt, pretty_dict, read_from_json, extract_filename_number, match_file_path
from programs.utils.arguments import process_args
from programs.utils.experiment_setup import get_default_config, setup_loss_instance
from programs.utils.device_setup import SetupDevice
from programs.core.data_manager import BatchSubset
from programs.core.model_manager import BestModelTracker
from programs.core.experiment_tasks import generate_samples


@log_execution_time
def main(args):

    if args.debug_on:
        set_logger_level(10)

    model_params = read_from_json(args.model_params_path, deserialize=True)

    if not model_params:
        logger.error(f"Missing saved model params from '{args.model_params_path}'")

    test_subset = BatchSubset(args.input_folder,'testing')
    test_loader = test_subset.get_dataloader(num_workers= max(args.num_tasks-1, 1))

    device = SetupDevice.setup_torch_device(args.num_tasks, 
                                            args.cpu_device_only, 
                                            args.gpu_device_list, 
                                            args.gpu_memory_fraction,
                                            args.random_seed)
    
    model_params['device'] = device
    model_params['data_transforms'] = test_subset.data_transforms
    model_params['generation_samples'] = args.generation_samples
    model_params['generation_noise'] = args.generation_noise
    model_params['random_seed'] = args.random_seed
    
    setup_loss_instance(model_params)

    logger.info(pretty_dict(model_params, label="MODEL_PARAMS"))

    model_config = get_default_config(model_params['model_key'])
    model_class = model_config['model_class']
    new_model = model_class(model_params).to(device)

    model_state_path = match_file_path(args.model_state_path)
    logger.debug(f"Found model_state_path: '{model_state_path}'")

    saved_model = BestModelTracker.load_model_state(new_model, model_state_path)

    epoch = extract_filename_number(args.model_state_path)
    generate_samples(saved_model, model_params, test_loader, epoch, args.output_folder)


if __name__ == "__main__":
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args(__file__)
        main(args)
    except Exception as e:
        logger.error(e)
