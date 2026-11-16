"""Flask Application Factory for GTSRB Traffic Sign Recognition Web Demo."""

import os

from flask import Flask

from webapp.config import MAX_UPLOAD_MB
from webapp.routes import api_bp, views_bp


def create_app() -> Flask:
    """Create and configure the modular Flask web application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_AS_ASCII"] = False

    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)
    return app
