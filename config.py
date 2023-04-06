from datetime import timedelta

SESSION_DAYS = 5


class Config:
    REMEMBER_COOKIE_DURATION = timedelta(days=SESSION_DAYS)
    RECAPTCHA_PUBLIC_KEY = '6LcT7UIfAAAAAKH-CbP8j1fk-iVLZTlJ6Rw6SXjG'
    RECAPTCHA_PRIVATE_KEY = '6LcT7UIfAAAAAKHe-sbJg5vpckN8ds9wCujdLI27'
    RECAPTCHA_DATA_ATTRS = {'bind': 'invisible-recaptcha',
                            'callback': 'onSubmit',
                            'size': 'invisible'}

class TestingConfig(Config):
    SECRET_KEY = 'test'
    TESTING = True
    DEBUG = True

class DevelopmentConfig(Config):
    SECRET_KEY = 'dev'
    DEBUG = True

class ProductionConfig(Config):
    SECRET_KEY = 'lV\xdcN\xd9\xd9\x07\x07\x9b\xf7o&\x9b\xe0\x84\x9at\x8b\x00b\x9f~\xa9j'


config = {
    'testing': TestingConfig,
    'development': DevelopmentConfig,
    'production': ProductionConfig
}
