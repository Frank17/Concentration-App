from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager

from config import config

manager = LoginManager()
manager.login_view = 'auth.login'


def create_app(config_name):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config[config_name])
    CORS(app)
    
    manager.init_app(app)
    manager.login_view = 'auth.login'

    from .auth import auth_bp
    app.register_blueprint(auth_bp)

    from .errors import errors_bp
    app.register_blueprint(errors_bp, url_prefix='/error')

    from .overview import overview_bp
    app.register_blueprint(overview_bp, url_prefix='/overview')

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
