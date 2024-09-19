from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import json
import time
import cv2
import numpy as np
from models.face_detector import process_frame  # 이 함수의 정의가 필요합니다.

app = Flask(__name__)

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
            'fr': True,
            'sleep': False,
            'yawn_count': 0
        }
    return render_template('student_monitor.html', student_name=student_name)

@app.route('/get_student_data/<student_name>')
def get_student_data(student_name):
    def event_stream():
        while True:
            time.sleep(2)  # 주기적으로 업데이트 (2초)
            student_data = students_data.get(student_name, {
                'fr': True,
                'sleep': False,
                'yawn_count': 0
            })
            yield f"data: {json.dumps(student_data)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/get_all_student_data')
def get_all_student_data():
    def event_stream():
        while True:
            time.sleep(2)  # 주기적으로 업데이트 (2초)
            yield f"data: {json.dumps(students_data)}\n\n"  # 전체 학생 데이터 전송
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/upload_frame/<student_name>', methods=['POST'])
def upload_frame(student_name):
    try:
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 이미지가 올바르게 읽혔는지 확인
        if frame is None:
            return jsonify({'error': '이미지 읽기 실패'}), 400

        # 이미지 처리
        fr, sleep, yawn_count, processed_frame = process_frame(frame, student_name)

        # 결과 저장
        students_data[student_name] = {
            'fr': fr,
            'sleep': sleep,
            'yawn_count': yawn_count
        }
        
        return jsonify({'fr': fr, 'sleep': sleep, 'yawn_count': yawn_count})  # 추가된 데이터 반환
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/teacher')
def teacher():
    return render_template('teacher_sse.html')

if __name__ == '__main__':
    app.run(threaded=True, port=5000, host="0.0.0.0")
