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

# 학생 모니터링용 프레임 생성기
def generate_frames(student_name, students_data):
    fr = False
    sleep = False
    yawn_count = 0
    timer = 0

    # 비디오 캡처
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        # 얼굴 감지
        rects = detector(frame, 0)
        
        if len(rects) == 0:
            fr = False
        else:
            fr = True
            sleep = False
            for rect in rects:
                shape = predictor(frame, rect)
                shape = np.matrix([[p.x, p.y] for p in shape.parts()])

                mar = mouth_ratio(shape)
                right_eye = shape[36:42]
                left_eye = shape[42:48]
                right_ear = eye_ratio(right_eye)
                left_ear = eye_ratio(left_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < 0.3 and mar < 0.5:
                    timer += 1
                    if timer >= 50:
                        sleep = True
                else:
                    timer = 0

                if mar > 0.70:
                    yawn_count += 1
                    time.sleep(3)

                frame = add_text(frame, f"EAR: {ear[0][0]:.2f}", (10, 30))
                frame = add_text(frame, f"하품 횟수: {yawn_count}", (10, 60))

        # 상태 업데이트
        students_data[student_name] = {'fr': fr, 'sleep': sleep, 'yawn_count': yawn_count}

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
