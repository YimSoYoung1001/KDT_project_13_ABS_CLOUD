from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy


# DB 관련 인스턴스 생성 
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # 설정 내용 로딩 
    # (config.py파일에서 읽어들여서 app.config 환경변수로 부르기 위함.)
    app.config.from_pyfile('./config.py')

    # ORM (object relational mapping) 즉, DB 초기화
    db.init_app(app) 
    migrate.init_app(app, db)

    # 테이블 클래스
    from . import models  # => 이 부분은 model.py를 환성한 이후 주석을 풀어줌 
    # migrate 객체가 model.py 파일을 참조하게 한다. 

    from flask import Blueprint
    # 여기에 직접 함수쓰는 대신에 blueprint 사용하도록 변경
    
    # 블루프린트
    from .views import main_views
    app.register_blueprint(main_views.bp)

    return app

