import os
import logging
import coloredlogs
import socket

DEBUG =0

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

