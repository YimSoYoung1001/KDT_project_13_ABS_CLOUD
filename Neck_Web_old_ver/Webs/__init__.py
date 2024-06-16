from flask import Flask


def create_app():
    app = Flask(__name__)

    from flask import Blueprint
    # 여기에 직접 함수쓰는 대신에 blueprint 사용하도록 변경
    from .views import main_views
    
    app.register_blueprint(main_views.bp)

    return app
