from flask import render_template
from . import errors_bp

# @errors_bp.app_errorhandler(400)
# def page_not_found(e):
#     return render_template('not_found.html', code=400), 400
    
@errors_bp.app_errorhandler(404)
def page_not_found(e):
    return render_template('not_found_404.html', code=404), 404

@errors_bp.app_errorhandler(502)
def bad_gateway(e):
    return render_template('bad_gateway_502.html', code=502), 502
    
@errors_bp.app_errorhandler(503)
def server_unavailable(e):
    return render_template('server_unavilable_503.html', code=503), 503


# Bad news...     We couldn't find this page!
# Maybe you want to navigate to...
# Login Signup What is Concentrate
# https://craftwork.design/your-purchases/
