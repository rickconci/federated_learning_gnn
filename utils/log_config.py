import logging


def create_logger(
    logger_name,
    verbose=False,
    logger_fmt="%(asctime)s - %(levelname)s - %(message)s",
):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(logger_fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if verbose:
        logger_level = logging.DEBUG
    else:
        logger_level = logging.INFO

    logger.setLevel(logger_level)
    return logger
