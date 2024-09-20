import dlib
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageDraw, ImageFont
import time

# 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

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


students_data = {}

# 학생 모니터링용 프레임 생성기
def process_frame(frame, student_name):
    # 학생 상태 정보 가져오기
    student_info = students_data.get(student_name, {'yawn_count': 0, 'timer': 0})
    yawn_count = student_info['yawn_count']
    timer = student_info['timer']  # 기존 타이머 값 가져오기
    fr = False
    sleep = False

    # 디버깅: 입력 프레임 크기 출력
    print("입력 프레임 크기:", frame.shape)

    # 얼굴 감지
    rects = detector(frame, 0)
    
    # 얼굴 감지 시도 로그
    print("얼굴 감지 시도 중...")
    if len(rects) == 0:
        fr = False  # 얼굴이 감지되지 않음
        print("얼굴 감지 실패")  # 디버깅 메시지
    else:
        fr = True  # 얼굴이 감지됨
        print(f"얼굴 감지 성공: {len(rects)}개 얼굴")

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
            print(f"EAR: {ear[0][0]:.2f}, MAR: {mar:.2f}, 현재 하품 횟수: {yawn_count}, 타이머: {timer}")

            if ear < 0.3:  # EAR 기준
                timer += 1
                if timer >= 50:  # 50 프레임 이상 졸음 감지
                    sleep = True  # 졸음 상태
                    print("졸음 감지")
            else:
                timer = 0
                sleep = False  # 졸음 상태 초기화

            if mar > 0.70:  # 하품 기준
                yawn_count += 1  # 하품 감지
                print("하품 감지")
                time.sleep(3)  # 잠시 대기 (하품 감지 후 지연)

            # 프레임에 텍스트 추가
            frame = add_text(frame, f"EAR: {ear[0][0]:.2f}", (10, 30))
            frame = add_text(frame, f"하품 횟수: {yawn_count}", (10, 60))
            frame = add_text(frame, f"졸음 상태: {'졸음' if sleep else '깨움'}", (10, 90))

    # 학생 데이터 업데이트
    students_data[student_name] = {
        'fr': fr,
        'sleep': sleep,
        'yawn_count': yawn_count,
        'timer': timer  # 타이머 값도 저장
    }

    # 최종 상태 로그
    print(f"학생: {student_name}, 얼굴 감지: {fr}, 졸음 상태: {sleep}, 하품 횟수: {yawn_count}, 타이머: {timer}")

    return fr, sleep, yawn_count, frame

