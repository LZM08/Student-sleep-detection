import dlib
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageDraw, ImageFont
import time

# 글로벌 변수 초기화
students_data = {}

# 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 텍스트 추가 함수
def add_text(img, text, position, color=(0, 255, 10), size=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # Malgun Gothic 폰트 사용
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 폰트 경로
    font = ImageFont.truetype(font_path, size, encoding="utf-8")

    draw.text(position, text, color, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 눈 비율 계산
def eye_ratio(eye):
    A = euclidean_distances(np.array(eye[1]), np.array(eye[5]))
    B = euclidean_distances(np.array(eye[2]), np.array(eye[4]))
    C = euclidean_distances(np.array(eye[0]), np.array(eye[3]))
    ratio = ((A + B) / 2.0) / C
    return ratio

# 입 비율 계산  
def mouth_ratio(shape):
    A = euclidean_distances(np.array(shape[50]), np.array(shape[58]))
    B = euclidean_distances(np.array(shape[51]), np.array(shape[57]))
    C = euclidean_distances(np.array(shape[52]), np.array(shape[56]))
    D = euclidean_distances(np.array(shape[48]), np.array(shape[54]))
    return ((A + B + C) / 3) / D

# 학생 모니터링용 프레임 생성기
def process_frame(frame, student_name):
    global students_data  # 글로벌 변수를 사용
    fr = False
    sleep = False
    timer = 0

    # 학생의 하품 횟수 가져오기
    if student_name not in students_data:
        students_data[student_name] = {'yawn_count': 0}  # 초기화
    yawn_count = students_data[student_name].get('yawn_count', 0)

    # 얼굴 감지
    rects = detector(frame, 0)
    
    if len(rects) == 0:
        fr = False  # 얼굴이 감지되지 않음
    else:
        fr = True  # 얼굴이 감지됨
        for rect in rects:
            shape = predictor(frame, rect)  # 얼굴 랜드마크 추출
            shape = np.matrix([[p.x, p.y] for p in shape.parts()])  # 좌표 변환

            mar = mouth_ratio(shape)  # 입 비율 계산
            right_eye = shape[36:42]  # 오른쪽 눈 랜드마크
            left_eye = shape[42:48]  # 왼쪽 눈 랜드마크
            right_ear = eye_ratio(right_eye)  # 오른쪽 눈 비율
            left_ear = eye_ratio(left_eye)  # 왼쪽 눈 비율
            ear = (left_ear + right_ear) / 2.0  # 평균 눈 비율

            # 졸음 및 하품 감지 로직
            if ear < 0.3:  # EAR 기준
                timer += 1
                if timer >= 50:  # 50 프레임 이상 졸음 감지
                    sleep = True  # 졸음 상태
            else:
                timer = 0

            if mar > 0.70:  # 하품 기준
                yawn_count += 1  # 하품 감지

            # 프레임에 텍스트 추가
            frame = add_text(frame, f"EAR: {ear[0][0]:.2f}", (10, 30))
            frame = add_text(frame, f"하품 횟수: {yawn_count}", (10, 60))
            frame = add_text(frame, f"졸음 상태: {'졸음' if sleep else '깨움'}", (10, 90))

    # 하품 카운트를 업데이트
    students_data[student_name]['yawn_count'] = yawn_count

    return fr, sleep, yawn_count, frame
