import os
import logging


# ----- Logger -----

class Colors:
    grey = "\x1b[0;37m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    reset = "\x1b[0m"
    blink_red = "\x1b[5m\x1b[1;31m"


class CustomFormatter(logging.Formatter):
    # For colored logs thanks to: https://github.com/herzog0/best_python_logger/blob/master/best_python_logger/core.py
    # Original StackOverflow post: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output,
    # answer by Sergey Pleshakov
    """Logging Formatter to add colors and count warning / errors"""
    def __init__(self, auto_colorized=True, datefmt: str = None):
        super(CustomFormatter, self).__init__(datefmt=datefmt)
        self.auto_colorized = auto_colorized
        self.datefmt = datefmt
        self.FORMATS = self.define_format()

    def define_format(self):
        # Levels
        # CRITICAL = 50
        # FATAL = CRITICAL
        # ERROR = 40
        # WARNING = 30
        # WARN = WARNING
        # INFO = 20
        # DEBUG = 10
        # NOTSET = 0

        format_prefix = f"{Colors.purple}%(asctime)s{Colors.reset} " \
                        f"{Colors.blue}%(name)s{Colors.reset} " \
                        f"{Colors.light_blue}(%(filename)s:%(lineno)d){Colors.reset} "

        format_suffix = "%(levelname)s - %(message)s"

        '%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s - %(funcName)s()] - %(message)s'

        return {
            logging.DEBUG: format_prefix + Colors.green + format_suffix + Colors.reset,
            logging.INFO: format_prefix + Colors.grey + format_suffix + Colors.reset,
            logging.WARNING: format_prefix + Colors.yellow + format_suffix + Colors.reset,
            logging.ERROR: format_prefix + Colors.red + format_suffix + Colors.reset,
            logging.CRITICAL: format_prefix + Colors.blink_red + format_suffix + Colors.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(logs_dir, add_file_handler=True, file_append_mode=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        fmt='%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)s] - %(levelname)s - %(message)s',
        datefmt='%Y/%m/%d - %H:%M:%S'
    )
    use_console_custom_formatter = True
    if use_console_custom_formatter:
        console_formatter = CustomFormatter(auto_colorized=True, datefmt='%Y/%m/%d - %H:%M:%S')
    else:
        console_formatter = file_formatter

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    file_mode = 'a' if file_append_mode else 'w'
    file_handler = logging.FileHandler(
        filename=os.path.join(logs_dir, 'logs.txt'),
        encoding='utf-8',
        mode=file_mode
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    if add_file_handler:
        logger.addHandler(file_handler)
    return logger


def log_message(message, rank=0, logger=None):
    if rank == 0:
        if logger is not None:
            logger.info(message)
        else:
            print(message)
