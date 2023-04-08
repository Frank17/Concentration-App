# import sys
# sys.path.append('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

from app import create_app

app = create_app('production')

if __name__ == '__main__':
    app.run()