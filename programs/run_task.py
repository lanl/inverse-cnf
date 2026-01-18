from programs.utils.logger_setup import init_shared_logger
logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
from programs.utils.common import os_path, pt, tb
from programs.utils.arguments import ap, process_args, executable_groups
from programs.utils.config_setup import validate_config_args, task_modules

def main():
    parser = ap.ArgumentParser(description="CLI program config runner")
    
    task_list = list(task_modules.keys())
    parser.add_argument('--task', '--task-name', '-t', dest="task_name", choices=task_list,
                    help=f"Task to process cli config arguments for choices: {task_list} | required")
    parser.add_argument('--config', '--config-file', '-c', dest="config_file", type=str, help="Input path to cli config file to process | optional")
    args, extra_args = parser.parse_known_args()

    if not args.config_file and not extra_args:
        raise ValueError("The --config-file argument is required when no extra arguments are passed.")

    if args.config_file and not os_path.exists(args.config_file):
        raise FileNotFoundError(f"CONFIG_FILE '{args.config_file}' does not exist")

    task_name = args.task_name
    config_file = args.config_file
    config_path = os_path.abspath(config_file) if config_file else None

    try:
        task_module = task_modules[task_name]
        if not hasattr(task_module, "main"):
            raise AttributeError(f"The module {task_name} does not have a 'main' function.")
        task_main = getattr(task_module, "main")

        if config_file:
            if not os_path.exists(config_path):
                raise FileNotFoundError(f"CONFIG_FILE '{config_path}' does not exist")
            task_args = validate_config_args(config_path, task_name)
        else:
            task_args = process_args(task_module.__file__, extra_args=extra_args)
    

        if 'multiprocess' in executable_groups[task_name]:
            pt.multiprocessing.set_sharing_strategy('file_system')

        task_main(task_args)

    except ModuleNotFoundError as e:
        logger.error(f"Cannot find [--task-name] '{task_name}'") 
    except Exception as e:
        logger.error(f"Program error from '{task_name}': {e}\n\n[Stack Trace]: {tb.format_exc()}")


if __name__ == "__main__":
    main()