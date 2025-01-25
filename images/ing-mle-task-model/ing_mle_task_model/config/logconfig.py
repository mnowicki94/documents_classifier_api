import logging
from datetime import datetime

def setup_logging(file_log_level=logging.DEBUG, console_log_level=logging.INFO):
    logging.basicConfig(level=file_log_level,
                        format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='logs/{:%Y_%m_%d_%H_%M}.log'.format(datetime.now()),
                        filemode='w')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)-12s - %(funcName)s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)

    logging.getLogger('').addHandler(console_handler)