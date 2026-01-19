from programs.utils.logger_setup import set_logger_level, init_shared_logger, log_execution_time
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.arguments import process_args
from programs.utils.common import pt, os_path
from programs.core.data_manager import PreprocessDataset, BatchSubset


@log_execution_time
def main(args):
    if args.debug_on:
        set_logger_level(10)
    try:
        prep_dataset = PreprocessDataset(args.dataset_file, 
                                            model_input_keys=args.model_input_keys, 
                                            model_target_keys=args.model_target_keys, 
                                            solver_input_keys=args.solver_input_keys, 
                                            unique_id_key=args.unique_id_key,
                                            subset_split=args.subset_split, 
                                            transform_method=args.transform_method,
                                            batch_size=args.batch_size,
                                            random_seed=args.random_seed,
                                            shuffle_on=args.shuffle_on,
                                            minmax_range= args.minmax_range,
                                            flatten_nested_keys=args.flatten_nested_keys)
    except Exception as e:
        raise Exception(f"PreprocessDataset failed: {e}")

    output_folder_path = os_path.abspath(args.output_folder)
    prep_dataset.save_dataset(output_folder_path)

    if args.debug_on:
        train_subset = BatchSubset(output_folder_path, "training")
        train_subset.inspect_batches(1)

        val_subset = BatchSubset(output_folder_path, "validation")
        val_subset.inspect_batches(1)
        
        test_subset = BatchSubset(output_folder_path, "testing")
        test_subset.inspect_batches(1)


if __name__ == "__main__":
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args(__file__)
        main(args)
    except Exception as e:
        logger.error(e)