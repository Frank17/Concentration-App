from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import credentials, initialize_app, db
from flask_login import UserMixin
from .. import manager

initialize_app(credentials.Certificate("FirebaseKey.json"),
                    {'databaseURL':
                     'https://study-focus-project-default-rtdb.'
                     'firebaseio.com/'}
              )


REF = db.reference('/Concentrateapp/Users')
id = max((i['id'] for i in REF.get().values()))


class EmailNotExistError(Exception):
    def __init__(self, email):
        self.message = f"Cannot find email '{email}' inside database"
        super().__init__(self.message)


class UserAlreadyExistError(Exception):
    pass


class UsernameAlreadyExistError(Exception):
    pass


def _get(email, get_info=True):
    for username, info in REF.get().items():
        if info['email'] == email:
            return info if get_info else username


def info_from_email(email):
    return _get(email)


def username_from_email(email):
    return _get(email, False)


def is_new_user(email):
    if email in {i['email'] for i in REF.get().values()}:
        raise UserAlreadyExistError


# def get_from_id(id: int):

def username_clash(username):
    return username in REF.get()
        
  
class OldUser(UserMixin):
    """
    TODO: Change email to id (PRIMARY KEY should not be changed)
    """
    def __init__(self, email, id=0):
        
        self.info = info_from_email(email)
        if self.info is None:
            raise EmailNotExistError(email)
        self.ref = REF
        self.email = email
        self.username = username_from_email(email)
        self.uid = self.info['id']

    
    def get_id(self):
        return self.email

    
    def get_info(self):
        return {
                    'username':      self.username,
                    'email':         self.email,
                    'id':            self.uid,
                    'password':      self.info['password'],
                    'block_list':    self.info['block_list'],
                    'status':        self.info['login_status']
               }

    
    def get_safe_info(self):
        return {
                    'username':      self.username,
                    'email':         self.email,
                    'block_list':    self.info['block_list']
               }

        
    def login(self, pwd):
        db_pwd = self.info['password']
        if check_password_hash(str(db_pwd), str(pwd)):
            self.update_info(status='true')
            return True
        return False


    def update_info(self, block_list=None, password=None, email=None, username=None, status=None):
        if block_list is None:            block_list = self.info['block_list']
        if password is None:              password = self.info['password']
        if status is None:                status = self.info['login_status']
        error_flag = False
        
        if email is None:
            email = self.info['email']
        else:
            self.email = email

        if username is None:
            username = self.username
        elif username_clash(username):
            error_flag = True
        else:
            self.ref.child(self.username).delete()
            self.username = username

        self.ref.update({
                self.username: {'block_list': block_list,
                                'email': email,
                                'password': password,
                                'id': self.uid,
                                'login_status': status}
        })

        self.info = info_from_email(email)

        if error_flag:
            raise UsernameAlreadyExistError
            
class NewUser(OldUser):
    def __init__(
        self,
        email: str,
        username: str,
        password: str,
        block_list: str = ''
    ):
        is_new_user(email)
        global id
        id += 1
        self.uid = id
        self.ref = REF
        self.email = email
        self.username = username
        self.ref.update({
                self.username: {'block_list': block_list,
                                'email': email,
                                'password': generate_password_hash(password),
                                'id': self.uid,
                                'login_status': True}
        })
        self.info = info_from_email(email)


@manager.user_loader
def load_user(email):
    return OldUser(email)
