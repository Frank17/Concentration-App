from flask import Blueprint

overview_bp = Blueprint('profile', __name__,
                        static_folder='static',
                        template_folder='templates')

from . import views
