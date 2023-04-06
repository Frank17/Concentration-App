from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, PasswordField, BooleanField
from wtforms.fields import EmailField
from wtforms.validators import InputRequired, Email, Length

class BaseUserForm(FlaskForm):
    email = EmailField('Email', validators=[
        InputRequired('Please provide an email!'),
        Email('The given email is invalid!')
    ])
    password = PasswordField('Password', validators=[
        InputRequired('Please provide a password!'),
        Length(7, 14, 'Given password must be at least 7 characters'
               'long and at most 14 characters long!')
    ])
    recaptcha = RecaptchaField()

    
class LoginForm(BaseUserForm):
    remember = BooleanField('Remember me')
    

class SignupForm(BaseUserForm):
    username = StringField('Username', validators=[
        InputRequired('Please provide a username!')
    ])
