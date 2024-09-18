from models.face_detector import generate_frames
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import json
import time
from datetime import datetime, timezone

app = Flask(__name__)

students_data = {}  # 학생들의 이름과 상태를 저장하는 딕셔너리

# 메인 페이지
@app.route('/') 
def index():
    return render_template('index.html')

# 학생 페이지
@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == 'POST':
        student_name = request.form['student_name']
        return redirect(url_for('student_monitor', student_name=student_name))
    return render_template('student.html')

# 학생 모니터링 페이지 (카메라 피드)
@app.route('/student_monitor/<student_name>')
def student_monitor(student_name):
    if student_name not in students_data:
        students_data[student_name] = {
            'fr': True,  # 임시 데이터
            'sleep': False,
            'yawn_count': 0
        }
    return render_template('student_monitor.html', student_name=student_name)

# 학생 데이터 SSE (이름별)
@app.route('/get_student_data/<student_name>')
def get_student_data(student_name):
    def event_stream():
        while True:
            time.sleep(2)  # 주기적으로 업데이트 (2초)
            student_data = students_data.get(student_name, {
                'fr': True,  # 임시 데이터
                'sleep': False,
                'yawn_count': 0
            })
            yield f"data: {json.dumps(student_data)}\n\n"  # JSON 형식으로 변환하여 전송
    return Response(event_stream(), mimetype="text/event-stream")

# 학생 데이터 SSE (전체)
@app.route('/get_all_student_data')
def get_all_student_data():
    def event_stream():
        while True:
            time.sleep(2)  # 주기적으로 업데이트 (2초)
            yield f"data: {json.dumps(students_data)}\n\n"  # JSON 형식으로 변환하여 전송
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/video_feed/<student_name>')
def video_feed(student_name):
    return Response(generate_frames(student_name, students_data),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 선생님 페이지
@app.route('/teacher')
def teacher():
    return render_template('teacher_sse.html')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
