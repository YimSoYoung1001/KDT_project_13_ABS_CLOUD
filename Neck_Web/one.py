import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import pyautogui
import math
import sqlite3
import datetime
import cv2

def pointDist(x1,y1,z1,x2,y2,z2):
    class Point3D:
        def __init__(self,x,y,z):
            self.x=x
            self.y=y
            self.z=z
    p1=Point3D(x=x1,y=y1,z=z1)
    p2=Point3D(x=x2,y=y2,z=z2)
    dist=math.sqrt(pow(p2.x-p1.x,2)+pow(p2.y-p1.y,2)+pow(p2.z-p1.z,2))
    # print(dist)
    return dist


# -----------------------------------------------------------------------------------------------
# A4 탐지 (세로로 들어야함 그래야 width로 저장됨)
# -----------------------------------------------------------------------------------------------
def find_A4():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("비디오 파일을 열 수 없습니다.")
            exit()
        # A4의 높이값 반환해주는 함수
        def detect_a4_paper(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 < aspect_ratio < 1.4 and w > 100 and h > 100:
                        return w
            return None
        while True:
            # 비디오 프레임 읽기
            success, image = cap.read()
            cv2.putText(image, 'If you want to stop, just press "q"', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            if not success:
                print("카메라를 찾을 수 없습니다.")
                # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
                continue
            # 프레임 화면에 표시
            cv2.imshow('camera', image) 
            ref_pixel_width = detect_a4_paper(image)
            if ref_pixel_width is not None:
                # 기준 물체의 실제 크기와 픽셀 크기 (예: A4 용지)
                ref_real_width = 21.0  # cm, A4 용지의 실제 가로 길이
                print(f"A4 paper width in pixels: {ref_pixel_width}")
                print(f"A4 paper real width: {ref_real_width} cm")
                break
            else: pass
            if cv2.waitKey(1) == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    except: pass
    return ref_real_width, ref_pixel_width


# -----------------------------------------------------------------------------------------------
# 웹캠, 영상 파일에서 자세 감지 (귀와 어깨 중앙선까지의 수직선의 거리 구하기)
# -----------------------------------------------------------------------------------------------

# SQLite 데이터베이스 연결
conn = sqlite3.connect('yimsoyoung.db')
c = conn.cursor()


def find_turtle(ref_real_width, ref_pixel_width):
    try : 
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("카메라를 찾을 수 없습니다.")
                    # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
                    continue

                # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                
                # x, y 는 [0.0, 1.0]으로 정규화됨
                # z 는 엉덩이 중간 지점의 깊이를 원점으로 하는 랜드마크의 깊이 나타냄 
                img_h, img_w, _ = image.shape
                focal_length = 1 * img_w

                # 픽셀 대 cm 비율 계산
                pixel_to_cm_ratio = ref_real_width / ref_pixel_width
                e_left_x = int(results.pose_landmarks.landmark[7].x * img_w)
                e_left_y = int(results.pose_landmarks.landmark[7].y * img_h)
                e_left_z = results.pose_landmarks.landmark[7].z * focal_length

                e_right_x = int(results.pose_landmarks.landmark[8].x * img_w)
                e_right_y = int(results.pose_landmarks.landmark[8].y * img_h)
                e_right_z = results.pose_landmarks.landmark[8].z * focal_length

                s_left_x = int(results.pose_landmarks.landmark[11].x * img_w)
                s_left_y = int(results.pose_landmarks.landmark[11].y * img_h)
                s_left_z = results.pose_landmarks.landmark[11].z * focal_length

                s_right_x = int(results.pose_landmarks.landmark[12].x * img_w)
                s_right_y = int(results.pose_landmarks.landmark[12].y * img_h)
                s_right_z = results.pose_landmarks.landmark[12].z * focal_length

                # 귀와 어깨 사이 거리값 구하기 (픽셀 단위)
                left_dist_pixel = pointDist(e_left_x, e_left_y, e_left_z, s_left_x, s_left_y, s_left_z)
                right_dist_pixel = pointDist(e_right_x, e_right_y, e_right_z, s_right_x, s_right_y, s_right_z)

                # 거리값을 cm 단위로 변환
                left_dist_cm = round(left_dist_pixel * pixel_to_cm_ratio, 3)
                right_dist_cm = round(right_dist_pixel * pixel_to_cm_ratio, 3)

                # 변환된 거리값을 이미지에 출력
                cv2.putText(
                    image,
                    text=f"left D {left_dist_cm} cm, right D {right_dist_cm} cm",
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=(0, 255, 0),
                    thickness=2
                )

                # 포즈 주석을 이미지 위에 그립니다.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 보기 편하게 이미지를 좌우 반전합니다.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                
                # 거북목 탐지 알림 발생 , database에 기록 추가
                if (left_dist_cm >= 2.5) or (right_dist_cm >= 2.5):
                    pyautogui.alert(text = 'Excuse me, you should sit up straight.',
                                    title = 'Knock Knock',
                                    button = 'got it',
                                    timeout=2000)   # 2초 후 time out
                    now = datetime.now()
                    occur = 'yes'
                    current_date = now.strftime("%Y-%m-%d")
                    current_time = now.strftime("%H:%M:%S")

                    # 데이터 삽입
                    c.execute("INSERT INTO events (occur, date, time) VALUES (?, ?, ?)", (occur, current_date, current_time))
                    conn.commit()
                    
                
                if cv2.waitKey(33) == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    except : pass