import os
import json
import argparse

import cv2
import numpy as np
import torch 
# from tqdm import tqdm
# from dotenv import load_dotenv

from flask import Flask, render_template, request


app = Flask(__name__, template_folder=r'C:\Users\HAMA\Desktop\openpose')  # 템플릿 폴더 경로 설정

def get_keypoints(frame, net, threshold):
    '''
    이미지, 추정치, 한계치를 입력 받고
    keypoints를 반환합니다.
    '''
    BODY_PARTS = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
        10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
        15: "LEye", 16: "REar", 17: "LEar"
    }
    # 입력 이미지의 사이즈 정의
    imageHeight, imageWidth, channel = frame.shape
    inHeight = 368
    inWidth = int((inHeight / imageHeight) * imageWidth)

    img_zeros = np.zeros((imageHeight, imageWidth, channel), np.uint8) #
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False) # 네트워크에 넣기 위한 전처리
    net.setInput(input_blob) # 전처리된 blob 네트워크에 입력

    # 결과 받아오기
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    keypoints_list = []
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = int((frame_width * point[0]) / out_width)
        y = int((frame_height * point[1]) / out_height)

        cv2.circle(img_zeros, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # [pointed]
        if prob > threshold:  
            cv2.putText(img_zeros, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            keypoints_list.append([float(x), float(y), prob, float(i)])
        # [not pointed]
        else:  
            cv2.circle(img_zeros, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img_zeros, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)
            keypoints_list.append([-1.0, -1.0, 0.0, -1.0])

    keypoints_json = {"keypoints": keypoints_list}
    return keypoints_json


def worker(input_path, output_path, model_path):
    '''input_path: 입력 파일 경로
       output_path: 출력 파일 경로
    '''
    def save_file(keypoints_dict, output_path):
        try:
            with open(output_path, "w") as f:
                json.dump(keypoints_dict, f)
        except Exception:
            return False
        return True
    
    model = None
    
    try:
        model = torch.load(model_path)
    except Exception:
        return False
    
    try:
        image = cv2.imread(input_path)
        keypoints_dict = get_keypoints(frame=image, net=model, threshold=0.15)
        save_file(keypoints_dict, output_path)
    except Exception:
        return False
    return True


@app.route("/")
def hello():
    return render_template("index.html")  # 'index.html' 렌더링

@app.route("/post", methods=['POST'])
def post():
    # 'input_path'와 'output_path' 필드를 추출, 기본값은 빈 문자열로 설정
    input_path = request.form.get('input_path', ' ')
    output_path = request.form.get('output_path', ' ') 
    model_path = request.form.get('model_path', ' ')

    worker(input_path, output_path, model_path)

if __name__ == "__main__":
    app.run(debug=True, port=80)  # Flask 애플리케이션 실행
