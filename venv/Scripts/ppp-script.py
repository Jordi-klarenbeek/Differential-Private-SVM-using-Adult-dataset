#!C:\Users\jordi\PycharmProjects\svm\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pypi-publisher==0.0.4','console_scripts','ppp'
__requires__ = 'pypi-publisher==0.0.4'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pypi-publisher==0.0.4', 'console_scripts', 'ppp')()
    )
