"""Blueprint registration for HTML views and REST prediction API."""

from .api import api_bp
from .views import views_bp

__all__ = ["api_bp", "views_bp"]
