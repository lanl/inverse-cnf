from programs.utils.logger_setup import get_logger
import argparse as ap
from programs.utils.common import os_path, np, cpu_count, match_file_path
from programs.utils.experiment_setup import (MODEL_KEYS, 
                                LOSS_TYPES, 
                                ACTIVATION_TYPES, 
                                MODEL_LOSS_COMPATIBILITY)
from programs.utils.solver_setup import SOLVER_KEYS, SOLVER_DEFAULT


executable_groups = {
    "preprocess_dataset": ["data", "multiprocess"],
    "train_evaluate": ["device", "multiprocess", "model", "experiment", "generate"],
    "generate_samples": ["device", "multiprocess", "state", "generate"],
    "analyze_model":['analysis'],
    "analyze_samples":['analysis', 'dataset', 'sample']
}

def extract_argument(args, name:str):
    field = getattr(args, name)
    if isinstance(field, (list, tuple)):
        if len(field) > 1:
            raise ValueError(f"{name.upper()} must be a single value when [--enable-tuning] is disabled.")
        return field[0]
    return field


def minmax_range_type(vals):
    if len(vals) != 2:
        raise ap.ArgumentTypeError("Provide exactly two values for --minmax-range.")
    min_val, max_val = map(float, vals)
    if min_val >= max_val:
        raise ap.ArgumentTypeError("The first value must be less than the second (min < max).")
    return [min_val, max_val]



def parse_multi_scale_weights(weights: list[float]|tuple[float]) -> tuple[float]:
    try:
        weights = tuple(float(w) for w in weights)  # Convert input strings to floats
        total = sum(weights)
        if not np.isclose(total, 1.0):
            normalized = tuple(w / total for w in weights)
            get_logger().warning(f"Normalized [--multi-scale-weights] to sum to 1.0: {weights} => {normalized}")
        return weights
    except ValueError:
        raise ap.ArgumentTypeError(f"Invalid values for --multi-scale-weights: {weights}. Expected numeric values")

def add_analysis_group(parser, exe_name):
    group = parser.add_argument_group("analyze_predictions options")

    group.add_argument('--output-folder', dest="output_folder", type=str, default="analysis", 
                    help="Output folder to save analysis to, saves to input directory if no path is specified | default: 'analysis'")

    if 'sample' in executable_groups[exe_name]:
        group.add_argument('--input-data-path', dest="input_data_path", type=str, required=True, 
                    help="Input file path to the input data file containing generated results | required")
        
        group.add_argument('--random-seed', dest="random_seed", type=int, default=None,
                        help="Random seed for getting random records from the data file  | default: None")
        
        group.add_argument('--num-records', dest="num_records", type=int, default=5,  
            help=f"Number of records to read from the data file | default: 5")
    else:
        group.add_argument('--input-folder', dest="input_folder", type=str,
            help="Input path to folder containing result data files | required")
    
        group.add_argument('--max-epoch', dest="max_epoch", type=int, default=None,
            help=f"Maximum epoch to for model performance plots | default: maximum epoch possible")


def check_analysis_args(args):
    if hasattr(args, 'input_folder'):
        input_folder= match_file_path(args.input_folder)
        if not os_path.isdir(input_folder):
            raise FileNotFoundError(f"INPUT_FOLDER '{args.input_folder}' does not exist")
    
    if hasattr(args, 'input_data-path'):
        input_data_path = match_file_path(args.input_data_path)
        if not os_path.isfile(input_data_path):
            raise FileNotFoundError(f"INPUT_DATA_PATH '{args.input_data_path}' does not exist")

    if hasattr(args, 'num_samples') and not (0 < args.num_samples < 1025):
        raise ValueError("NUM_SAMPLES must be a INT between [9, 1025]")
    
    if hasattr(args, 'max_epoch') and args.max_epoch and args.max_epoch < 1:
        raise ValueError("MAX_EPOCH must be a INT greater than 1")
    
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"OUTPUT_FOLDER must have between [1, 255] chars")


def add_device_group(parser):
    group = parser.add_argument_group('pytorch device options')
    group.add_argument('--gpu-device-list', dest='gpu_device_list', type=int, nargs='+', default=[0], 
                    help='Specify which GPU(s) to use; Provide multiple GPUs as space-separated values, e.g., "0 1" | default: 0 (if CUDA is available)')
    group.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction', type=float, default=0.5,
                    help='Fraction of GPU memory to allocate per process | default: 0.5 (if CUDA is available)')
    group.add_argument('--cpu-device-only', dest="cpu_device_only", action='store_true',
                    help="PyTorch device can only use default CPU; Overrides other device options | default: Off")


def check_device_args(args):
    if len(args.gpu_device_list) < 1:
        raise ValueError("GPU_DEVICE_LIST must have at least 1 device, the default gpu is '0'")
    if isinstance(args.gpu_device_list, int):
        args.gpu_device_list = [args.gpu_device_list]
    if not (0.0 < args.gpu_memory_fraction < 1.01) :
        raise ValueError("GPU_MEMORY_FRACTION must have at least 1 device, the default gpu is '0'")
    return args


def add_data_group(parser):
    group = parser.add_argument_group("data processing options")

    group.add_argument('--dataset-file', dest="dataset_file", type=str, required=True, 
                    help="Input path/to/<input> dataset file | required")

    group.add_argument('--output-folder', dest="output_folder", type=str, default="shape_batches", 
                    help="Output path/to/folder to save batches to | required")

    group.add_argument('--batch-size', dest="batch_size", type=int, default=2, 
                    help="Batch size for loading dataset into pytorch model | default: 2")

    group.add_argument('--batch-shuffle', dest="shuffle_on", action='store_true',
                        help="Enables shuffling training batches during data loading | default: Off")
    
    group.add_argument('--subset-split', dest="subset_split", nargs='+', type=float, default=[0.7,0.1,0.2],
                    help="Ratios to split the original dataset into [Train,Validate,Test] or [Train,Validate] subsets | default: [0.7, 0.1, 0.2]")

    group.add_argument('--random-seed', dest="random_seed", type=int, default=None,
                    help="Random seed for dataset Train-Validate-Test split | default: None")

    group.add_argument('--transform-method', dest="transform_method", type=str, choices=["minmax", "standard"], 
                    help="Type of data transformation method to perform: ['minmax', 'standard']; Only applies to [--model-inputs] and [--model-targets] | default:  None")
    
    group.add_argument('--minmax-range', dest="minmax_range", nargs=2, type=float, default=[-1.0, 1.0],
                        help="Target min and max values for [--transform-method] 'minmax'  | default: (-1.0, 1.0)")

    group.add_argument('--model-input-keys', dest="model_input_keys", nargs='+', type=str, required=True,
                    help="List of dataset keys for the model input (X). | required")

    group.add_argument('--model-target-keys', dest="model_target_keys",nargs='+', type=str, required=True,
                    help="List of dataset keys for the model target (Y) | required")

    group.add_argument('--solver-input-keys', dest="solver_input_keys", nargs='+', type=str, required=False,
                    help="List of dataset keys for physics solver inputs (not provided to the model)")

    group.add_argument('--unique-id-key', dest="unique_id_key", type=str,
        help=("Name of the dataset column to use as a unique record identifier for lookups or reproducibility. "
            "This is typically a simulation seed or config ID that uniquely defines each sample. "
            "If not provided, the dataset index will be used as a fallback."
        )
    )

    group.add_argument('--flatten-nested-keys', dest='flatten_nested_keys', action='store_true', help="Enables flattening parent keys for nested HDF5 data | default: Off")
    

def check_data_args(args):
    if not os_path.isfile(args.dataset_file):
        raise FileNotFoundError(f"DATASET_FILE '{args.dataset_file}' does not exist")
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"OUTPUT_FOLDER must have between [1, 255] chars")
    if not (0 < args.batch_size < 4097):
        raise ValueError("BATCH_SIZE must be a INT between [1, 4097]")

    if not (0 < len(args.model_input_keys)):
        raise ValueError("MODEL_INPUT_KEYS must have at least 1 data key in it")
    if not (0 < len(args.model_target_keys)):
        raise ValueError("MODEL_TARGET_KEYS must have at least 1 data key in it")

    if len(args.subset_split) not in range(2,4) or any(r <= 0 for r in args.subset_split) or not np.isclose(np.sum(args.subset_split), 1.0):
        raise ValueError("SUBSET_SPLIT must have 2-3 ratios that add up to 1.0")
    
    if args.transform_method == "minmax":
        if args.minmax_range is None:
            raise ValueError("MINMAX_RANGE is required when 'minmax' is set in [--normalize-config]")
        elif args.minmax_range[0] >= args.minmax_range[1]:
            raise ValueError("MINMAX_RANGE must be formatted as (min, max), where min < max")
        args.minmax_range = tuple(args.minmax_range)
    else:
        args.minmax_range = None
        
    return args


def add_model_group(parser: ap.ArgumentParser):
    group = parser.add_argument_group("model default options")

    group.add_argument('--model-key', dest="model_key", metavar="MODEL_KEY", 
                        type=str, choices=MODEL_KEYS, default=MODEL_KEYS[0],
                        help=f"Model key to train from choices: {MODEL_KEYS}")

    group.add_argument('--solver-key', dest="solver_key", metavar="SOLVER_TYPE", 
                        type=str, choices=SOLVER_KEYS, default=SOLVER_DEFAULT,
                        help=f"Solver key for physics loss from choices: {SOLVER_KEYS}. | default: {SOLVER_DEFAULT}")
    
    group.add_argument('--enable-tuning', dest="tuning_on", action='store_true',
                        help="Enable model hyperparameter tuning with Optuna. | default: Off")

    group.add_argument('--activation-function', dest="activation_function", nargs='*', metavar="ACTIVATION_FUNCTION", type=str, choices=ACTIVATION_TYPES, default=None, 
                    help=f"Choice for the activation function; Type is List[str] when [--enable-tuning] flag, otherwise str;\nChoices: {ACTIVATION_TYPES} | default: model default")

    exc_loss_group = group.add_mutually_exclusive_group(required=False)
    exc_loss_group.add_argument('--loss-function', dest="loss_function", nargs='*', metavar="LOSS_FUNCTION", type=str, choices=LOSS_TYPES, default=None,
                    help=f"Choice for the pixel loss function; Type is List[str] when [--enable-tuning] flag, otherwise str;\nChoices: {LOSS_TYPES} | default: model default")
    
    exc_loss_group.add_argument('--hybrid-loss-functions', dest="hybrid_loss_functions", nargs='+', metavar="HYBRID_LOSS_FUNCTION", choices=LOSS_TYPES, type=str, default=None,
                    help=f"Choice of 2-5 pixel loss functions to combine into one; Set [--hybrid-loss-weights] to modify weights;\nChoices: {LOSS_TYPES} | default: model default")

    group.add_argument('--hybrid-loss-weights', dest="hybrid_loss_weights", nargs='+', metavar="HYBRID_LOSS_WEIGHTS", type=float, default=[0.5, 0.5],  
                    help="Weights for the hybrid loss function; Weights are applied in the same order as [--hybrid-loss-functions]; | default: [0.5, 0.5]")
    
    group.add_argument('--learn-rate', dest="learn_rate", nargs='*', type=float, default=[1e-4],  
                    help="Learning rate for model optimizer. Type is List[float] when [--enable-tuning] flag, otherwise float. | default: 1e-4")
    
    group.add_argument('--block-networks', dest="block_networks", nargs='+', type=int, default=[6],  
                    help=f"Number of network blocks. Type is List[int] when [--enable-tuning] flag, otherwise int. | default: 6")
    
    group.add_argument('--hidden-layers', dest="hidden_layers", nargs='+', type=int, default=[3],  
                    help="Number of network layers; For block-Type is List[int] when [--enable-tuning] flag, otherwise int. | default: 3")
    
    group.add_argument('--num-neurons', dest="num_neurons", nargs='+', type=int, default=[128], 
                    help="Number of neurons per layer. Type is List[int] with [--enable-tuning] flag, otherwise int. | default: 128")

    group.add_argument('--conv-kernel', dest="conv_kernel", type=int, default=3, 
                    help="Kernel size for convolutional layers, ignored if model is not a CNN type. | default: 3")

    group.add_argument('--conv-stride', dest="conv_stride", type=int, default=1, 
                    help="Stride for convolutional layers (controls downsampling), ignored if model is not a CNN type. | default: 1")

    group.add_argument('--multi-scale-kernel','--ms-ssim-kernel', dest="multi_scale_kernel", type=int, default=None,
                        help="Explicit kernel MS-SSIM Loss; Must be odd and in range [3, 15] | | default: None | required: if 'ms-ssim' in [--hybrid-loss-functions | --loss-function]" )
    
    group.add_argument('--multi-scale-weights','--ms-ssim-weights', dest="multi_scale_weights", nargs='+', type=float, default=None,
                        help="Weights per scale for MS-SSIM Loss; Number of weights must equal (kernel+1)//2 | default: None | required: if 'ms-ssim' in [--hybrid-loss-functions | --loss-function]" )

    group.add_argument('--affine-log-bounds', dest="affine_log_bounds", nargs=2, type=float, default=(0.0, 2.0),
        help="Affine Coupling log transformation bounds; Format as (min_value, max_value), where 0.0 <= min_value < max_value <= 10.0;  | default: (0.0, 2.0)")

    group.add_argument('--affine-log-transform', dest="affine_log_transform", metavar="AFFINE_LOG_TRANSFORM",
        type=str, choices=["squared_tanh", "scaled_tanh", "sigmoid", "softplus", "clamp"], default='squared_tanh', 
        help=(
            "Affine Coupling Log transformation method — controls how log values are scaled:\n"
            "  - 'squared_tanh': Squared tanh, output in [0, 1], non-negative.\n"
            "  - 'scaled_tanh': Tanh scaled by abs(max_log) scale preserving symmetry.\n"
            "  - 'sigmoid': Sigmoid scaled to [min_log, max_log].\n"
            "  - 'softplus': Softplus transformation, smooth and positive.\n"
            "  - 'clamp': Clamps log values to [min_log, max_log].\n"
            "default: 'squared_tanh'"
        )
    )
    
    group.add_argument('--beta-schedule-mode', dest="beta_schedule_mode", metavar="BETA_MODE", 
        type=str, choices=["linear", "cosine", "exp", "constant"], default="cosine",
        help=(
            "Beta scheduling mode — controls how β (pixel loss weight) decays over time:\n"
            "  - 'linear': Linearly decays β from 1.0 to 0.0\n"
            "  - 'cosine': Smooth cosine decay from 1.0 to 0.0\n"
            "  - 'exp': Exponentially decays β (sharp drop early)\n"
            "  - 'constant': Keeps β fixed at 1.0 (no decay)\n"
            "Use to balance reconstruction vs. latent losses during training. | default: cosine"
        )
    )

    group.add_argument('--beta-schedule-epochs', dest="beta_schedule_epochs", nargs='+', type=int, default=[200],
        help="Number of epochs over which β (pixel-loss weight) decays from <min_value> to <min_value> (ranges set for [--beta-schedule-mode]). "
            "Used to gradually reduce pixel-loss during training. | default: 200")

    group.add_argument('--beta-warmup-epochs', dest="beta_warmup_epochs", type=float, default=0.1,
        help=("Beta scheduling warmup - controls total epochs where β is fixed at the max_value before decay starts. | default: 0.1 (10%%)"
            "- If warmup >= 1: Fixed number of epochs (integer) \n"
            "- If 0.0 <= warmup < 1.0: Fraction of total [--beta-schedule-epochs] (float)")
    )

    group.add_argument('--beta-value-range', dest="beta_value_range", type=float, nargs=2, default=(0.0, 1.0),
        help="Beta decay value range; Format as (min_value, max_value), where 0.0 <= min_value < max_value <= 1.0;  | default: (0.0, 1.0)")


    group.add_argument('--projection-method', dest="projection_method", metavar="PROJECTION_METHOD", type=str, nargs='+', choices=["linear", "mlp", "resnet"], default=["linear"],
        help=(
            "Method for transforming latent Z into condition Y;\n"
            "  - 'linear': 1x1 Convolutional mapping\n"
            "  - 'mlp': Multi-layer perceptron prediction head\n"
            "  - 'resnet': Residual Network prediction head\n"
            "default: 'linear'"
        )
    )

    group.add_argument('--projection-activation', dest="projection_activation", metavar="PROJECTION_ACTIVATION", type=str, nargs='+', choices=ACTIVATION_TYPES, default=["identity"], 
        help=f"Activation for when [--projection-method] is not set to 'linear'; Choices: {ACTIVATION_TYPES} | default: 'identity'")


def check_model_args(args):

    compatible_loss_list = MODEL_LOSS_COMPATIBILITY[args.model_key]

    if args.tuning_on:
        if any(not (0 < n < 2049) for n in args.num_neurons):
            raise ValueError("NUM_NEURONS must be LIST[INT] with values between [1, 2048]")
    
        if any(not (0 < n < 129) for n in args.hidden_layers):
            raise ValueError("HIDDEN_LAYERS must be LIST[INT] with values between [1, 128]")

        if any(not (1E-9 <= n <= 0.99) for n in args.learn_rate):
            raise ValueError("LEARN_RATE must be LIST[FLOAT] with values between [1E-9, 0.99]")

        if any(not(0 < n < 129) for n in args.block_networks):
            raise ValueError("BLOCK_NETWORKS must be a LIST[INT] with values between [1, 128]")

        if any(not(0 <= n <= args.num_epochs) for n in args.beta_schedule_epochs):
            raise ValueError(f"BETA_SCHEDULE_EPOCHS must be LIST[INT] with values between [0, {args.num_epochs}]")

        if args.loss_function:
            if isinstance(args.loss_function, (list, tuple)):
                for loss in args.loss_function:
                    if loss not in compatible_loss_list:
                        raise ValueError(f"LOSS_FUNCTION {args.loss_function} must contain compatible losses: {compatible_loss_list}")
            elif isinstance(args.loss_function, str):
                if args.loss_function not in compatible_loss_list:
                    raise ValueError(f"LOSS_FUNCTION must be a compatible loss function: {compatible_loss_list}")
    else:
        args.num_neurons = extract_argument(args, "num_neurons")
        if not (0 < args.num_neurons < 2049):
            raise ValueError("NUM_NEURONS must be a INT between [1, 2048]")

        args.hidden_layers = extract_argument(args, "hidden_layers")
        if not (0 < args.hidden_layers < 129):
            raise ValueError("HIDDEN_LAYERS must be a INT between [1, 128]")

        args.learn_rate = extract_argument(args, "learn_rate")
        if not (1E-9 <= args.learn_rate <= 0.99):
            raise ValueError("LEARN_RATE must be a FLOAT between [1E-9, 0.99]")

        args.block_networks = extract_argument(args, "block_networks")
        if not (0 < args.block_networks < 129):
            raise ValueError("BLOCK_NETWORKS must be a INT between [1, 128]")
        
        args.beta_schedule_epochs = extract_argument(args, "beta_schedule_epochs")
        if not (0 <= args.beta_schedule_epochs <= args.num_epochs):
            raise ValueError(f"BETA_SCHEDULE_EPOCHS must be INT between [0, {args.num_epochs}]")
    
        args.activation_function = extract_argument(args, "activation_function")

        if args.loss_function:
            args.loss_function = extract_argument(args, "loss_function")
            if args.loss_function not in compatible_loss_list:
                raise ValueError(f"LOSS_FUNCTION must be a compatible loss function: {compatible_loss_list}")

        args.projection_method = extract_argument(args, "projection_method")
        args.projection_activation = extract_argument(args, "projection_activation")

    beta_min, beta_max = args.beta_value_range
    if not (0.0 <= beta_min < beta_max <= 1.0):
        raise ValueError("BETA_VALUE_RANGE must be 2 floats (min_val, max_val), where (0.0 <= min_val < max_val <= 1.0).")
    beta_epochs = min(args.beta_schedule_epochs) if isinstance(args.beta_schedule_epochs, (list, tuple)) else args.beta_schedule_epochs
    beta_warmup = float(args.beta_warmup_epochs)

    if not (0 <= beta_warmup <= beta_epochs):
        raise ValueError(f"BETA_WARMUP_EPOCHS must be a FLOAT between [0, 1.0) OR a INT between [1, {beta_epochs}]")
    if not beta_warmup.is_integer() and beta_warmup > 1.0:
        raise ValueError(f"BETA_WARMUP_EPOCHS must be an INT between [1, {beta_epochs}] when greater than or equal to 1.0")

    if isinstance(args.hybrid_loss_functions, (list, tuple)):
        loss_count = len(args.hybrid_loss_functions)

        if not (2 <= loss_count <= 5):
            raise ValueError(f"HYBRID_LOSS_FUNCTION must contain between 2 and 5 loss types. Got: {loss_count}")

        if len(set(args.hybrid_loss_functions)) !=  loss_count:
            raise ValueError("HYBRID_LOSS_FUNCTION cannot contain duplicate loss types")
    
        if len(args.hybrid_loss_weights) != loss_count:
            raise ValueError("HYBRID_LOSS_WEIGHTS must have 1 weight per loss function")
        
        if not np.isclose(np.sum(args.hybrid_loss_weights), 1.0):
            raise ValueError(f"HYBRID_LOSS_WEIGHTS must sum to 1.0")

        for loss_key in args.hybrid_loss_functions:
            if loss_key not in compatible_loss_list:
                raise ValueError(f"Loss '{loss_key}' is not in the list of compatible loss functions: {compatible_loss_list}")


    handle_msssim_args = 'ms-ssim' in compatible_loss_list and ((args.loss_function and 'ms-ssim' in args.loss_function) 
                                            or (args.hybrid_loss_functions and 'ms-ssim' in args.hybrid_loss_functions))
    if handle_msssim_args:
        if not args.multi_scale_kernel or not args.multi_scale_weights:
            raise ValueError("[--multi-scale-kernel] and [--multi-scale-weights] are required if 'ms-ssim' is a loss function")
        
        if args.multi_scale_kernel % 2 == 0 or not (3 <= args.multi_scale_kernel <= 15):
            raise ValueError("MULTI_SCALE_KERNEL must be an odd INT in range [3, 15]")
        
        required_scales = (args.multi_scale_kernel+1)//2
        if len(args.multi_scale_weights) != required_scales:
            raise ValueError(f"MULTI_SCALE_WEIGHTS must have weights for (kernel+1)//2 = {required_scales} total scales")
        
        args.multi_scale_weights = parse_multi_scale_weights(args.multi_scale_weights)
    else:
        args.multi_scale_kernel = None
        args.multi_scale_weights = None

    if not (0 < args.conv_kernel < 2049):
        raise ValueError("CONV_KERNEL must be an INT between [3, 2048 (assumed max)]")
    if not (0 < args.conv_stride < 2049):
        raise ValueError("CONV_STRIDE must be an INT between  [1, 2048 (assumed max)]")

    if hasattr(args ,"affine_log_transform") and isinstance(args.affine_log_transform, str):
        log_lb, log_ub = args.affine_log_bounds
        if not (0.0 <= log_lb < log_ub):
            raise ValueError("AFFINE_LOG_BOUNDS must be 2 floats (min_val, max_val), where (0.0 <= min_val < max_val).")

    return args

def add_experiment_group(parser):
    group = parser.add_argument_group("model training/validation/test loop options")
    group.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where results batches are saved | required")

    group.add_argument('--output-folder', dest="output_folder", type=str, required=True, 
                help="Output path/to/directory to save results | required")

    group.add_argument('--num-epochs', dest="num_epochs",  type=int, default=200,  
                    help="Number of total training epochs | default: 200")
    
    group.add_argument('--checkpoint-frequency', '--checkpoint-freq', dest="checkpoint_frequency", type=int, default=None,
            help="Enables saving model states by a frequency. If not specified or zero only the best and last model states are saved. | default: 0")

    group.add_argument('--validation-frequency', '--validation-freq', dest="validation_frequency", type=int, default=2,
            help="Epoch frequency to run validation phase | default: 2")

    group.add_argument('--testing-frequency', '--testing-freq', dest="testing_frequency", type=int, default=0,
            help="Epoch frequency to run testing phase. If not specified or zero, testing is performed only once with the best model state. | default: 0")

    group.add_argument('--testing-samples', dest="testing_samples", type=int, default=3,
                    help="Number of samples to generate during testing phase during training/validation loop. More samples will increase overhead during training/validation loop. Default: 3")
    
    group.add_argument('--plot-debug-images', dest="plot_debug_images_on", action='store_true',
                    help="Enable plotting random images from training, validation, and testing phases for debugging purposes. Plots occur every validation epoch. Default: Off")

    group.add_argument('--enable-earlystop', dest="earlystop_on", action='store_true',
                    help="Enables optuna to stop epochs early when avg validation loss stops improving | default: Off")

    group.add_argument('--pruner-option', dest="pruner_option", type=str, choices=["median", "threshold"],
                        help="Specify the pruner strategy to use for early stopping. Choices are 'median' (median-based pruning) or 'threshold' (threshold-based pruning). | default: 'median'.")

    group.add_argument('--threshold-upper', dest="threshold_upper", type=float, default=0.0,
                        help="Upper bound threshold for early stopping with threshold pruning. Prune trials where the objective (loss) exceeds this value. | default: 0.0")

    group.add_argument('--min-delta', dest="min_delta", type=float, default=100.0,
                        help="Minimum change in the objective value (validation loss) to be considered as an improvement for early stopping. If no improvement greater than this value is observed, the trial is pruned. | default: 1000.0")

    group.add_argument('--epoch-patience', dest="epoch_patience", type=int, default=20,
                        help="Number of epochs to wait without improvement in validation loss before triggering early stopping. | default: 20")

    group.add_argument('--trial-patience', dest="trial_patience", type=int, default=2,
                        help="Number of trials to wait without improvement in validation loss before triggering early stopping for the whole trial. | default: 2")



def check_experiment_args(args):
    if not os_path.isdir(args.input_folder):
        raise FileNotFoundError(f"INPUT_FOLDER '{args.input_folder}' does not exist")
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"OUTPUT_FOLDER '{args.output_folder}'  must have a length between [1, 255]")
    if not (0 < args.num_epochs < 1E9):
        raise ValueError("NUM_EPOCH must be a INT between [1, 1E9]")
    if args.checkpoint_frequency is not None and not (0 <= args.checkpoint_frequency <= args.num_epochs):
        raise ValueError("CHECKPOINT_FREQ must be a INT between [0, NUM_EPOCH]")
    if not (1 <= args.validation_frequency <= args.num_epochs):
        raise ValueError("VALIDATION_FREQUENCY must be a INT between [1, NUM_EPOCH]")
    if not (0 <= args.testing_frequency <= args.num_epochs):
        raise ValueError("TESTING_FREQUENCY must be a INT between [0, NUM_EPOCH]")
    if not (0 < args.testing_samples <= 1000):
        raise ValueError("TESTING_SAMPLES must be a INT between [1, 1000]")
    
    if args.tuning_on and args.earlystop_on:
        if not (-1e9 <= args.threshold_upper <= 1e9):
            raise ValueError("THRESHOLD_UPPER must be a FLOAT between [-1e9, 1e9]")
        if not (-1e9 <= args.min_delta <= 1e9):
            raise ValueError("MIN_DELTA must be a FLOAT between [-1e9, 1e9]")
        if not (0 <= args.epoch_patience <= args.num_epochs):
            raise ValueError(f"EPOCH_PATIENCE must be a INT between [0, {args.num_epochs}]")
        if not (1 <= args.trial_patience <= 200):
            raise ValueError(f"TRIAL_PATIENCE must be an INT between [1, 200]")


def add_state_group(parser):
    group = parser.add_argument_group("model test options")
    group.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where results batches are saved | required")
    group.add_argument('--output-folder', dest="output_folder", type=str, required=True, 
                help="Output path/to/directory to save results | required")
    group.add_argument('--model-state-path', dest="model_state_path", type=str, required=True, 
                help="Input file path to the saved model state file | required")
    group.add_argument('--model-params-path', dest="model_params_path", type=str, required=True, 
                help="Input file path to where model params are saved to | required")


def check_state_args(args):
    if not os_path.isdir(args.input_folder):
        raise FileNotFoundError(f"INPUT_FOLDER '{args.input_folder}' does not exist")
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"OUTPUT_FOLDER '{args.output_folder}' must have a length between [1, 255]")
    model_state_path = match_file_path(args.model_state_path)
    if not os_path.isfile(model_state_path):
        raise FileNotFoundError(f"MODEL_STATE_PATH '{args.model_state_path}' does not exist")
    model_params_path = match_file_path(args.model_params_path)
    if not os_path.isfile(model_params_path):
        raise FileNotFoundError(f"MODEL_PARAMS_PATH '{args.model_params_path}' does not exist")


def add_generate_group(parser: ap.ArgumentParser):
    group = parser.add_argument_group("generation options")
    group.add_argument('--generation-samples', dest="generation_samples", type=int, default=100,
                        help="Number of samples to generate during inference with a saved model state. Default: 100")
    group.add_argument('--generation-limit', dest="generation_limit", type=int, default=1,
                        help="Maximum samples to generate at a given time to avoid OOM errors. Default: 1")
    group.add_argument('--generation-noise', dest="generation_noise", type=float, default=1.0,
                        help="Scaling factor for noise added to the generated samples. Higher values increase noise and diversity in the outputs. Default: 1.0.")
    group.add_argument('--random-seed', dest="random_seed", type=int,
                            help="Random RNG seed for reproducing sample generation. default: None")

def check_generate_args(args):
    if not (0 < args.generation_samples <= 1e6):
        raise ValueError(f"GENERATION_SAMPLES must be INT between [1, 1e6]")
    if not (0 < args.generation_noise <= 10):
        raise ValueError(f"GENERATION_NOISE must be FLOAT between [0.0, 10.0]")
    if not (1 <= args.generation_limit <= 1e6):
        raise ValueError(f"GENERATION_LIMIT must be INT between [1, 1e6]")

def add_multiprocess_group(parser):
    group = parser.add_argument_group('multi-process options')
    group.add_argument('--ntasks', dest="num_tasks", type=int, default=1, 
                    help="Number of tasks (cpu cores) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")

def check_multiprocess_args(args):
    if not (1 <= args.num_tasks < cpu_count()):
        raise ValueError(f"NUM_TASKS must be a INT between [1, {cpu_count()} - 1]")


def check_args(args, program_name):
    if "multiprocess" in executable_groups[program_name]:
        check_multiprocess_args(args)
    if "data" in executable_groups[program_name]:
        args = check_data_args(args)
    if "experiment" in executable_groups[program_name]:
        check_experiment_args(args)
    if "state" in executable_groups[program_name]:
        check_state_args(args)
    if "generate" in executable_groups[program_name]:
        check_generate_args(args)
    if "analysis" in executable_groups[program_name]:
        check_analysis_args(args)
    if "device" in executable_groups[program_name]:
        args = check_device_args(args)
    if "model" in executable_groups[program_name]:
        args = check_model_args(args)
    return args


def log_args(args):
    args_msg = f"\nParsed Arguments:\n"
    for arg, value in vars(args).items():
        args_msg += f"[{arg}]: {value}\n"
    get_logger().info(args_msg)


def process_args(exe_file, show_args=True, extra_args=None, parser_only=False):
    program_name = os_path.basename(exe_file).replace('.py', '')
    parser = ap.ArgumentParser(description="Inverse Generative Project", formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('--debug', '-d', dest='debug_on', action='store_true', help="Enables debug option and verbose printing | default: Off")
    
    if "device" in executable_groups[program_name]:
        add_device_group(parser)
    if "multiprocess" in executable_groups[program_name]:
        add_multiprocess_group(parser)
    if "data" in executable_groups[program_name]:
        add_data_group(parser)
    if "model" in executable_groups[program_name]:
        add_model_group(parser)
    if "experiment" in executable_groups[program_name]:
        add_experiment_group(parser)
    if "generate" in executable_groups[program_name]:
        add_generate_group(parser)
    if "state" in executable_groups[program_name]:
        add_state_group(parser)
    if "analysis" in executable_groups[program_name]:
        add_analysis_group(parser, program_name)

    if parser_only:
        return parser
    
    args = parser.parse_args(extra_args) if extra_args else parser.parse_args()
    args = check_args(args, program_name)
    if show_args:
        log_args(args)

    return args