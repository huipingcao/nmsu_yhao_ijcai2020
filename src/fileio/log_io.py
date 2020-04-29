import logging
import sys
from datetime import datetime
import os
#import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#logging.getLogger("tensorflow").setLevel(logging.INFO)

##
# Try not to initialize many loggers at the same run
# def init_logging(log_file=''):
#     if log_file != '':
#         log_file = datetime.now().strftime(log_file + '_%Y_%m_%d_%H_%M.log')
#     log_format = "%(levelname)s %(asctime)-15s [%(lineno)d] %(funcName)s: %(message)s"
#     if log_file == '':
#         logging.basicConfig(format=log_format, level=logging.INFO, stream=sys.stdout) 
#     else:
#         logging.basicConfig(filename=log_file, filemode='w', format=log_format, level=logging.INFO)
#     logger = logging.getLogger()
#     #logger = tf.get_logger()
#     return logger


def init_logging(log_file='', name='log_name', level=logging.DEBUG):
    if len(log_file) == 0:
        return init_logging(log_file)
    """Function setup as many loggers as you want"""
    if log_file != '':
        log_file = datetime.now().strftime(log_file + '_%Y_%m_%d_%H_%M.log')
    formatter = logging.Formatter('%(levelname)s %(asctime)-15s [%(lineno)d] %(funcName)s: %(message)s')
    handler = logging.FileHandler(log_file, 'w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    #logger = tf.get_logger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


if __name__ == '__main__':
    #log_file = "/home/ivan/Research/projects/yhao_cnn_varying/src/python/test"
    #init_logging(log_file)
    #setup_logger(log_file)
    #logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logger = setup_logger('')
    #logging.debug('This message should appear on the console')
    logger.info('So should this')
    #logging.warning('And this, too')
