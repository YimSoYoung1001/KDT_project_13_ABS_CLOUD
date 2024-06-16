from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import os

import os

## 다양한 DBMS URI - SQLITE
BASE_DIR = os.path.dirname(__file__)
DB_NAME_SQLITE = 'app.db'

## 다양한 DBMS URI
DB_SQLITE_URI = f"sqlite:///{os.path.join(BASE_DIR, DB_NAME_SQLITE)}"      # 내컴
DB_MYSQL_URI = 'mysql+pymysql://root:0000@localhost:3306/testdb'           # 원격
                                # id : pw
                                
# DB_MARAI_URI = 'mariadb+mariadb://root:root@localhost:0000/testdb'
# DB_POST_URi = "postgresql+pg8000://scott:tiger@localhost/test"

# 사용할 DBMS 설정 / SQLALCHEMY_시작 변수명 고정 
SQLALCHEMY_DATABASE_URI = DB_MYSQL_URI   # 데이터베이스 접속 주소
SQLALCHEMY_TRACK_MODIFICATIONS = False   # SQLAlchemy의 이벤트 처리하는 옵션 (현재 불필요하므로 비활성화)
# 위 둘은 환경 변수라서 변수명을 반드시 저걸로 해주어야 함
