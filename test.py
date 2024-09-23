import time
import numpy as np
import dlib
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageDraw, ImageFont

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

# 눈 그리기
def draw_eye(eye):
    eye_contour = cv2.convexHull(eye)
    cv2.drawContours(frame, [eye_contour], -1, (0, 255, 0), 1)

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

# 타이머 변수 및 하품 카운터
timer = 0
yawn_count = 0  # 하품 횟수 카운터
stop = 0
# 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Student-sleep-detection\models\shape_predictor_68_face_landmarks.dat")

# 비디오 캡처
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    # 얼굴 감지
    rects = detector(frame, 0)
    
    if len(rects) == 0:  # 얼굴이 인식되지 않으면 텍스트 표시
        frame = add_text(frame, "얼굴 인식 실패", (250, 250), color=(0, 0, 255))
    else:
        for rect in rects:
            shape = predictor(frame, rect)
            shape = np.matrix([[p.x, p.y] for p in shape.parts()])
            
            # 입과 눈 비율 계산
            mar = mouth_ratio(shape)
            right_eye = shape[36:42]
            left_eye = shape[42:48]
            right_ear = eye_ratio(right_eye)
            left_ear = eye_ratio(left_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # 눈이 감긴 경우
            if ear < 0.3 and mar < 0.5:
                timer += 1
                if timer >= 50:
                    frame = add_text(frame, "wake up!!!", (250, 250))
            else:
                timer = 0
            
            # 하품 감지 (입이 크게 벌어지면 하품으로 간주)
            if mar > 0.70:
                yawn_count += 1
                frame = add_text(frame, "하품 감지됨", (250, 300))
                time.sleep(3)
                
                
            
            # 눈과 하품 정보 표시
            draw_eye(left_eye)
            draw_eye(right_eye)
            
            info = "EAR: {:.2f}".format(ear[0][0])
            frame = add_text(frame, info, (0, 30))
            
            yawn_info = "하품 횟수: {}".format(yawn_count)
            frame = add_text(frame, yawn_info, (0, 70))
    
    # 프레임을 창에 표시
    cv2.imshow("Frame", frame)
    
    # ESC키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
