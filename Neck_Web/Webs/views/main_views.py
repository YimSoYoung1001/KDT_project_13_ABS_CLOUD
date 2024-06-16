from flask import Blueprint, render_template, request
import os, datetime, cv2, pyautogui, math, torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import mediapipe as mp         # 이거를 하니까 url은 안뜨는데 연결은 됨

from one import find_A4, find_turtle, pointDist
from Webs.models import Test


# -----------------------------------------------------------------------------------------------
# 모델 준비
# -----------------------------------------------------------------------------------------------
# 저장된 모델 돌릴 클래스
class MyResNet(torch.nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.resnet = models.resnet18()
    def forward(self, x):
        return self.resnet(x)

# 모델 인스턴스 생성
model = models.resnet18()
# 전결합층 변경
model.fc = torch.nn.Linear(in_features = 512, out_features = 2)

# 저장된 모델의 가중치 로딩
model_file = "./Webs/static/model/sign_total_100000.pth"
model.load_state_dict(torch.load(model_file))

# 모델이 학습된 형태의 이미지로 변화
preprocessing = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



# ===============================================================================================



# Blueprint 인스턴스 생성 
bp = Blueprint('main', 
               __name__, 
               template_folder='templates',
               url_prefix='/')


@bp.route('')
def first_page():
    return render_template(template_name_or_list='index.html')

@bp.route('1p')
def page_01():
    return render_template(template_name_or_list='page_01.html')


@bp.route('2p')
def page_02():
    return render_template(template_name_or_list='page_02.html')


@bp.route('3p')
def page_03():
    record = Test.query.order_by(Test.date.desc())

    # return render_template(template_name_or_list='page_03.html')
    return render_template('page_03.html', t_list = record)

@bp.route('turtle_result', methods = ['POST','GET'])
def check_turtle():
    if request.method == 'POST':
        # -----------------------------------------
        # 업로드된 이미지를 로컬에 저장
        # -----------------------------------------
        # 이미지 파일 경로
        dir = f"./Webs/static/img/"
        suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        file = request.files['file']
        save_dir = os.path.join(dir, suffix + file.filename)
        
        # 이미지 저장
        file.save(save_dir)
        
        # 불러올 이미지 경로
        open_dir = '../static/img/' + suffix + file.filename
        img_tag = f"<img src = '{open_dir}'>"
        
        open_img_dir = f'./Webs/static/img/{suffix + file.filename}'
        img = Image.open(open_img_dir)

        p_img = preprocessing(img)

        # -----------------------------------------
        # 모델 시연
        # -----------------------------------------
        model.eval()

        with torch.no_grad():
            p_img = p_img.unsqueeze(dim = 0)
            output = model(p_img)
            result = torch.argmax(output, dim=1).item()
        if result == 0 : diagnosis = 'negative'
        else : diagnosis = 'positive'

        return (f"<h1>img</h1><br> {img_tag}<br><br> <h1>result : {result}</h1><br> <h1>diagnosis : {diagnosis}</h1>")


@bp.route('turtle_cctv', methods=['POST','GET'])
def cctv_turtle():
    result = request.form['answer']

    if result == 'yes': 
        try:
            ref_real_width, ref_pixel_width = find_A4()
            return find_turtle(ref_real_width, ref_pixel_width)
        except: return ("<h1>종료됩니다.</h1><h1>main page로 가세요</h1> <a href = 'http://127.0.0.1:5000/' class = 'button'>main page</a>")
    else :
        return render_template('index.html')