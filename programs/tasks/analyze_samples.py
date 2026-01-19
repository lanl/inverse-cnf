from programs.utils.logger_setup import init_shared_logger, set_logger_level
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.arguments import process_args
from programs.utils.common import os_path, create_folder, match_file_path, extract_filename_number, read_from_hdf5
from programs.viz.sample_plots import get_data_ranges, plot_generated_sample_scores, plot_generated_images, plot_reconstructed_images

def main(args):

    if args.debug_on:
        set_logger_level(10)

    num_records = args.num_records
    random_seed = args.random_seed

    input_data_path = match_file_path(args.input_data_path)
    logger.debug(f"Found input_data_path: '{args.input_data_path}'")

    if os_path.exists(args.output_folder):
        output_folder = args.output_folder
    else:
        output_folder = os_path.dirname(input_data_path)

    sample_records = read_from_hdf5(input_data_path, sample_size=num_records, random_seed=random_seed)

    file_name = os_path.basename(input_data_path)

    epoch = extract_filename_number(file_name)

    plot_path = create_folder(os_path.join(output_folder, "sample_plots", f"epoch_{epoch}"))

    if "testing" in file_name:
        phase = "testing"
    elif "validation" in file_name:
        phase = "validation"
    elif "training" in file_name:
        phase = "training"
    
    if phase == "testing":
        plot_generated_images(sample_records, epoch, plot_path)
        plot_generated_sample_scores(sample_records, epoch, plot_path)
    else:
        plot_reconstructed_images(sample_records, epoch, phase, plot_path)

if __name__ == "__main__":
    try:
        args = process_args(__file__)
        main(args)
    except Exception as e:
        logger.error(e)