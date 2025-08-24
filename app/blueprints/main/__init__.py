from flask import Blueprint


main_bp = Blueprint('main', __name__)

# Import routes to register view functions
from . import routes  # noqa: E402,F401