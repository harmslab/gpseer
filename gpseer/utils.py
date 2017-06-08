import os
from datetime import datetime

def add_datetime_to_filename(filepath):
    """Inserts a datetime at the end of a filename, before the extension."""
    name, ext = os.path.splitext(filepath)
    # Get current date/time in isoformat
    t = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    return "{:}-{:}{:}".format(*[name, t, ext])
