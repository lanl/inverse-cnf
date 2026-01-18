import logging
from pathlib import Path
from os import getppid
import pandas as pd
from datetime import datetime
from functools import wraps
from inspect import stack
from sys import stderr, stdout, exit

class LoggerHandler:
    def __init__(self, logger):
        self.logger = logger

    def _log_message(self, log_func, msg, stack_level=2, exc_info=False):

        frame = stack()[stack_level]
        filename = Path(frame.filename).name
        function_name = frame.function
        lineno = frame.lineno
        log_func(f"[{filename}:{function_name}:{lineno}] || {msg}", exc_info=exc_info)

    def critical(self, msg: str = "unknown critical msg"):
        self._log_message(self.logger.critical, msg, stack_level=2, exc_info=True)
        exit(-1)

    def error(self, msg: str = "unknown error msg"):
        self._log_message(self.logger.error, msg, stack_level=2, exc_info=True)
        exit(-1)

    def warning(self, msg: str = "unknown warning msg"):
        self._log_message(self.logger.warning, msg, stack_level=2)

    def info(self, msg: str = "unknown info msg"):
        self._log_message(self.logger.info, msg, stack_level=2)

    def debug(self, msg: str = "unknown debug msg"):
        self._log_message(self.logger.debug, msg, stack_level=2)


class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.logger = None
            cls._instance.handler = None
            cls._instance.handler_obj = None
        return cls._instance

    def setup_logger(self, program_file, log_stdout=False, log_stderr=True):
        if self.logger is not None:
            return self.logger

        module_name = Path(program_file).stem

        unique_id = f"{module_name}_ppid{getppid()}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
        log_file = f"{unique_id}.log"
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_path = logs_dir / log_file

        logger = logging.getLogger("shared_logger")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(logs_path)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] || [%(levelname)s] || [%(filename)s:%(funcName)s:%(lineno)d] || %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if log_stdout:
                stdout_handler = logging.StreamHandler(stdout)
                stdout_handler.setLevel(logging.INFO)
                stdout_formatter = logging.Formatter('[%(asctime)s] || [%(levelname)s] || [%(filename)s:%(funcName)s:%(lineno)d] || %(message)s')
                stdout_handler.setFormatter(stdout_formatter)
                logger.addHandler(stdout_handler)

            if log_stderr:
                stderr_handler = logging.StreamHandler(stderr)
                stderr_handler.setLevel(logging.WARNING)
                stderr_formatter = logging.Formatter('[%(asctime)s] || [%(levelname)s] || [%(filename)s:%(funcName)s:%(lineno)d] || %(message)s')
                stderr_handler.setFormatter(stderr_formatter)
                logger.addHandler(stderr_handler)

        self.logger = logger
        self.handler_obj = LoggerHandler(self.logger)
        return self.logger

    def get_handler(self):
        if self.logger is None:
            raise ValueError("Logger is not initialized. Call setup_logger first.")
        return self.handler_obj


def get_logger():
    logger_instance = SingletonLogger()
    return logger_instance.get_handler()


def init_shared_logger(program_file, log_stdout=False, log_stderr=True):
    logger_instance = SingletonLogger()
    logger = logger_instance.setup_logger(program_file, log_stdout, log_stderr)
    handler = logger_instance.get_handler()
    return handler

def set_logger_level(level:int=20):
    if level in logging._levelToName:
        logger = get_logger()
        logger.logger.setLevel(level)
        for handler in logger.logger.handlers:
            if (isinstance(handler, logging.FileHandler) or handler.stream is stdout):
                handler.setLevel(level)
            logger.info(f"Logger handler: {handler}")
        
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()
        logger.info(f"Starting clock for {func.__name__} at {start_time}")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"Ending clock for {func.__name__} at {end_time}")
        elapsed_time = (end_time - start_time).total_seconds()
        logger.info(f"Elapsed time for {func.__name__} is {elapsed_time:.6f} seconds")
        return result
    return wrapper