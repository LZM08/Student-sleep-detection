from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import json
import time
import cv2
import numpy as np
from flask_cors import CORS
import base64
import dlib
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageDraw, ImageFont
import time


# 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Student-sleep-detection\models\shape_predictor_68_face_landmarks.dat")

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
fr = None
sleep = None

# 학생 모니터링용 프레임 생성기
def process_frame(frame, student_name):
    

    # 학생 상태 정보 가져오기
    student_info = students_data.get(student_name, {'yawn_count': 0, 'timer': 0})
    yawn_count = student_info['yawn_count']
    timer = student_info['timer']  # 기존 타이머 값 가져오기


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
            
            students_data[student_name] = {
                'fr': fr,
                'sleep': sleep,
                'yawn_count': yawn_count,
                'timer': timer  # 타이머 값도 저장
            }
            print(f"학생: {student_name}, 얼굴 감지: {fr}, 졸음 상태: {sleep}, 하품 횟수: {yawn_count}, 타이머: {timer}")

    # 학생 데이터 업데이트
    

    # 최종 상태 로그
    

    return fr, sleep, yawn_count, timer


app = Flask(__name__)
CORS(app)
students_data = {}  # 학생들의 이름과 상태를 저장하는 딕셔너리

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == 'POST':
        student_name = request.form['student_name']
        return redirect(url_for('student_monitor', student_name=student_name))
    return render_template('student.html')

@app.route('/student_monitor/<student_name>')
def student_monitor(student_name):
    if student_name not in students_data:
        students_data[student_name] = {
            'fr': False,
            'sleep': None,
            'yawn_count': 0,
            'timer': 0
        }
    return render_template('student_monitor.html', student_name=student_name)





@app.route('/get_student_data/<student_name>')
def get_student_data(student_name):
    def event_stream():
        while True:
            time.sleep(2)  # 주기적으로 업데이트 (2초)
            student_data = students_data.get(student_name, {
                'fr': False,
                'sleep': None,
                'yawn_count': 0,
                'timer': 0
            })
            yield f"data: {json.dumps(student_data)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")








@app.route('/get_all_student_data')
def get_all_student_data():
    def event_stream():
        while True:
            time.sleep(2)  # 2초마다 업데이트
            data_to_send = json.dumps(students_data)
            print("Sending data:", data_to_send)  # 보내는 데이터 로그
            yield f"data: {data_to_send}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")




@app.route('/upload_frame/<student_name>', methods=['POST'])
def upload_frame(student_name):
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 전송되지 않았습니다.'}), 400
        
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '이미지 읽기 실패'}), 400

        # 이미지 처리
        fr, sleep, yawn_count, timer = process_frame(frame, student_name)  # processed_frame은 필요 없음

        # 결과 저장
        students_data[student_name] = {
            'fr': fr,
            'sleep': sleep,
            'yawn_count': yawn_count,
            'timer': timer
        }

        # 학생 데이터 업데이트 확인
        print(f"Updated student data: {students_data[student_name]}")

        return jsonify({
            'fr': fr,
            'sleep': sleep,
            'yawn_count': yawn_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400





@app.route('/teacher')
def teacher():  
    return render_template('teacher_sse.html')

if __name__ == '__main__':
    app.run(threaded=True, port=5000, host="0.0.0.0")
