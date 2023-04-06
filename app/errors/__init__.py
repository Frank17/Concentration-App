from flask import Blueprint

errors_bp = Blueprint('errors', __name__,
                      static_folder='static',
                      template_folder='templates')

from . import views
