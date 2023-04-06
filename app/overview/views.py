from flask import render_template, redirect, flash, url_for
from . import overview_bp


@overview_bp.route('/introduction')
def introduction():
    return render_template('base.html')
