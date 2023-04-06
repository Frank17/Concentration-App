from flask import request, render_template, redirect, url_for, flash, session, jsonify
from flask_login import login_user, login_required, fresh_login_required, logout_user, current_user
from werkzeug.datastructures import MultiDict
from . import auth_bp, REDIRECTION_SECONDS
from .models import OldUser, EmailNotExistError, NewUser, UserAlreadyExistError, UsernameAlreadyExistError
from .forms import LoginForm, SignupForm


@auth_bp.before_request
def disable_session():
    session.permanent = False


@auth_bp.route('/')
def homepage():
    return 'You\'re at homepage now'


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    1. Get user's input email and password
    2. Verify their identity and redirect to a new page
        - If the user is new, redirect them to the signup page
        - If the user enters the wrong pwd, ask them to enter another one
        - If the user succeeds, redirect them to their profile page
    """

    form = LoginForm()

    if form.validate_on_submit():
        rform = request.form
        email, pwd = form.email.data, form.password.data

        try:
            user = OldUser(email)
        except EmailNotExistError:
            flash('New Account Detected')
            return redirect(url_for('auth.signup'))

        if not user.login(pwd):
            session['login_form'] = rform
            flash('Wrong password')
            return redirect(request.path)

        login_user(user, remember=('remember' in rform))
        return redirect(url_for('auth.profile'))

    if wpform := session.get('login_form'):
        session.pop('login_form')
        form = LoginForm(MultiDict(wpform))
        form.validate()

    return render_template('login.html', form=form)


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    rd = False
    username = form.username.data
    email, pwd = form.email.data, form.password.data

    if form.validate_on_submit():
        try:
            user = NewUser(email, username, pwd)
        except UserAlreadyExistError:
            session['login_form'] = request.form
            session['signup_form'] = request.form

            flash('Account already exist! You will be redirected to the login page'\
                  'in a few seconds.')

            return redirect(request.path)

        login_user(user)
        return redirect(url_for('auth.profile'))

    if rdform := session.get('signup_form'):
        session.pop('signup_form')
        form = SignupForm(MultiDict(rdform))
        form.validate()
        rd = True

    return render_template('signup.html',
                           form=form,
                           redirect_url=url_for('auth.login') if rd else 0,
                           sec=REDIRECTION_SECONDS)


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have successfully logged yourself out.')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        rform = request.form
        new_blist = rform.get('blist_update')
        uname, email, blist = rform.get('uname'), rform.get(
            'email'), rform.get('blist')
        
        if blist is not None:
            """
            We basically need to transfer the block list from Flask to JS, where it will be
            interpreted as the websites that the user wishes to block
            """
            return jsonify(blist)

        
        if email:
            session['_user_id'] = email

        try:
            current_user.update_info(username=uname,
                                     email=email,
                                     block_list=blist)
        except UsernameAlreadyExistError:
            flash('Sorry, the username is already taken!')

        return redirect(request.path)
        

    return render_template('profile.html',
                           user_info=current_user.get_safe_info())
