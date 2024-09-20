#!/bin/bash

# 시스템 패키지 설치
apt-get update && apt-get install -y cmake g++ make libgl1-mesa-glx

# Python 패키지 설치
pip install -r requirements.txt
