# 모듈 로딩 --------------------------------------------------
from Webs import db
from datetime import datetime
    
class Test(db.Model):
    __tablename__ = 'NewTable'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    occur = db.Column(db.String(100))
    date = db.Column(db.String(100))
    time = db.Column(db.String(100))
