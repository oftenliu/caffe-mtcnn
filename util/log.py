
import logging
def get_logger(name=None):
  """return a logger
  """
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
  sh.setFormatter(formatter)
  logger.addHandler(sh)
  return logger