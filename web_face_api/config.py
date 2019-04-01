# Project config
import logging
import os
import yaml

_logger = logging.getLogger(__name__)
_config = {}

def init(config_file='_config.yml'):
    if not os.path.exists(config_file):
        try:
            os.system(r'touch %s' % config_file)
        except OSError:
            _logger.error("Could not create config file: %s" % config_file)
            raise
    
    # Read config
    global _config
    _logger.debug("Trying to read config file: %s" % config_file)
    try:
        with open(config_file, "r") as f:
            _config = yaml.safe_load(f)
    except OSError:
        _logger.error("Can't open config file: '%s'", config_file)
        raise

def get_path(items, default=None):
    global _config
    curConfig = _config
    if isinstance(items, str) and items[0] == '/':
        items = items.split('/')[1:]
    for key in items:
        if key in curConfig:
            curConfig = curConfig[key]
        else:
            _logger.warning("/%s not specified in profile, defaulting to "
                            "'%s'", '/'.join(items), default)
            return default
    return curConfig

def has_path(items):
    global _config
    curConfig = _config
    if isinstance(items, str) and items[0] == '/':
        items = items.split('/')[1:]
    for key in items:
        if key in curConfig:
            curConfig = curConfig[key]
        else:
            return False
    return True


def has(item):
    return item in _config

def get(item='', default=None):
    if not item:
        return _config
    if item[0] == '/':
        return get_path(item, default)
    try:
        return _config[item]
    except KeyError:
        _logger.warning("%s not specified in profile, defaulting to '%s'",
                        item, default)
        return default
    