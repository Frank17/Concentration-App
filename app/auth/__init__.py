from flask import Blueprint

REDIRECTION_SECONDS = 3

auth_bp = Blueprint('auth', __name__,
                    static_folder='static',
                    template_folder='templates')

from . import views


# block list
# username
# password
# email
