from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import json
import time
import cv2
import numpy as np
import base64
import dlib
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ImageDraw, ImageFont
from flask_socketio import SocketIO, emit
import threading

# 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

students_data = {}
timeout_threshold = 10 

# 오래된 학생 정보 제거
def cleanup_old_students():
    while True:
        time.sleep(5)
        current_time = time.time()
        students_to_delete = []
        for student_name, data in students_data.items():
            if current_time - data['last_update'] > timeout_threshold:
                students_to_delete.append(student_name)

        for student_name in students_to_delete:
            del students_data[student_name]
            print(f"학생 데이터 삭제됨: {student_name}")

# 백그라운드에서 데이터 정리 스레드 실행
cleanup_thread = threading.Thread(target=cleanup_old_students)
cleanup_thread.daemon = True
cleanup_thread.start()

def eye_ratio(eye):
    A = euclidean_distances(np.array(eye[1]), np.array(eye[5]))
    B = euclidean_distances(np.array(eye[2]), np.array(eye[4]))
    C = euclidean_distances(np.array(eye[0]), np.array(eye[3]))
    return ((A + B) / 2.0) / C

def mouth_ratio(shape):
    A = euclidean_distances(np.array(shape[50]), np.array(shape[58]))
    B = euclidean_distances(np.array(shape[51]), np.array(shape[57]))
    C = euclidean_distances(np.array(shape[52]), np.array(shape[56]))
    D = euclidean_distances(np.array(shape[48]), np.array(shape[54]))
    return ((A + B + C) / 3) / D

def process_frame(frame, student_name):
    fr = None
    sleep = None
    yawn_count = 0
    student_info = students_data.get(student_name, {'yawn_count': 0, 'timer': 0, 'last_update': time.time()})
    yawn_count = student_info['yawn_count']
    timer = student_info['timer']
    rects = detector(frame, 0)

    if len(rects) == 0:
        fr = False
    else:
        fr = True
        for rect in rects:
            shape = predictor(frame, rect)
            shape = np.matrix([[p.x, p.y] for p in shape.parts()])
            mar = mouth_ratio(shape)
            right_eye = shape[36:42]
            left_eye = shape[42:48]
            right_ear = eye_ratio(right_eye)
            left_ear = eye_ratio(left_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.23:
                timer += 1
                if timer >= 20:
                    sleep = True
            else:
                timer = 0
                sleep = False

            if mar > 0.70:
                yawn_count += 1
                time.sleep(3)

    students_data[student_name] = {
        'fr': fr,
        'sleep': sleep if sleep is not None else False,
        'yawn_count': yawn_count,
        'timer': timer,
        'last_update': time.time() 
    }
    return fr, students_data[student_name]['sleep'], yawn_count, timer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == 'POST':
        student_name = request.form['student_name']
        return redirect(url_for('student_monitor', student_name=student_name))
    return render_template('student.html')

@app.route('/get_all_student_data')
def get_all_student_data():
    def event_stream():
        while True:
            time.sleep(0.5)
            data_to_send = json.dumps(students_data)
            yield f"data: {data_to_send}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/student_monitor/<student_name>')
def student_monitor(student_name):
    if student_name not in students_data:
        students_data[student_name] = {
            'fr': False,
            'sleep': None,
            'yawn_count': 0,
            'timer': 0,
            'last_update': time.time()
        }
    return render_template('student_monitor.html', student_name=student_name)

@app.route('/get_student_data/<student_name>')
def get_student_data(student_name):
    def event_stream():
        while True:
            time.sleep(0.5)
            student_data = students_data.get(student_name, {
                'fr': False,
                'sleep': None,
                'yawn_count': 0,
                'timer': 0,
                'last_update': time.time()
            })
            yield f"data: {json.dumps(student_data)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@socketio.on('image')
def handle_image(data):
    npimg = np.frombuffer(base64.b64decode(data['image']), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    student_name = data['student_name']
    fr, sleep, yawn_count, timer = process_frame(frame, student_name)
    emit('response', {
        'fr': fr,
        'sleep': sleep,
        'yawn_count': yawn_count
    })

@app.route('/teacher')
def teacher():
    return render_template('teacher_sse.html')

# 새로운 라우트: 하품 횟수 초기화
@app.route('/reset_yawn_count', methods=['POST'])
def reset_yawn_count():
    for student in students_data:
        students_data[student]['yawn_count'] = 0
    return jsonify({"status": "success", "message": "하품 횟수가 초기화되었습니다."})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000)