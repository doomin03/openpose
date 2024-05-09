from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS

import json

import cv2
import numpy as np

import boto3
import boto3
import sys
from logging import Logger





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

# # 이 파일에는
# /data/Projects/eot/jhm/segment-anything/assets/masks1.png

# s3_object_url = s3://대충url:/버킷명/폴더/폴더/파일명
# # aws s3 저장소
def worker(input_path, output_path):
    '''
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로
    '''
    # 모델이 활성화되어야 한다.
    model = None
    def save_file(keypoints_dict, output_path):
        try:
            with open(output_path, "w") as f:
                json.dump(keypoints_dict, f)
        except Exception:
            return False
        return True
    
    model = None
    
    # try:
    #     model = torch.load(model_path)
    # except Exception:
    #     return False
    
    try:
        image = cv2.imread(input_path)
        keypoints_dict = get_keypoints(frame=image, net=model, threshold=0.15)
        save_file(keypoints_dict, output_path)
    except Exception:
        return False
    return True


class S3_Task:
    def __init__(self, access_key, secret_key, bucket_name):
        self.ACCESS_KEY = access_key
        self.SECRET_KEY = secret_key
        self.BUCKET_NAME = bucket_name
        self.connect()
    
    def connect(self) -> True|False:
        try:
            self.client = boto3.client(
                's3',
                aws_access_key_id=self.ACCESS_KEY,
                aws_secret_access_key=self.SECRET_KEY)
        except:
            return False
        return True
    
    def get_object_info_list(self, s3_prefix) -> list|bool:
        try:
            object_info_list = self.client.list_objects_v2(
            Bucket=self.BUCKET_NAME,
            Prefix=s3_prefix,)
        except:
            return False
        return object_info_list
        
    def download_s3_object(self, s3_path:str, local_path:str) -> True|False:
        try:
            self.client.download_file(self.BUCKET_NAME, s3_path, local_path) # 객체 다운로드
        except:
            return False
        return True

    def upload_s3_object(self, s3_path, local_path):
        try:
            self.client.upload_file(self.BUCKET_NAME, s3_path, local_path)
        except:
            return False
        return True
    


app = Flask(__name__)
CORS(app)
api = Api(app=app,version='1.0', title="openpose API")

#호출 규칙
input_model = api.model('Input',{
    # #download_s3_object메소드를 통해 다운로드할 위치
    # "input_path" : fields.String(required=True),
    # #worker 통해 오픈포즈를 저장하고 upload_s3_object 통해 결과를 배출
    # "output_path" : fields.String(required=True),
    
    # #s3 권한 키
    # "access_key" : fields.String(required=True),
    # #s3 시크릿 키
    # "secret_key" : fields.String(required=True),
    # #s3 버킷 이름
    # "bucket_name" : fields.String(required=True),
    
    # #item 객체 주소???? 
    # "s3_prefix" : fields.String(required=True),
    
    # "access_key": fields.String(required=True),
    # "secret_key": fields.String(required=True),
    # "bucket_name": fields.String(required=True),
    "아이템 키": fields.String(required=True),  # S3에서 다운로드할 객체의 키
    "s3_upload_url": fields.String(required=True),  # 처리된 파일을 저장할 경로
})

# 입력된 값 + 코드 단에서 https://docs.aws.amazon.com/ko_kr/{12345}/latest/userguide/Welcome.html
# s3_download_url = 'https://docs.aws.amazon.com/ko_kr/{12345}/latest/userguide/Welcome.html'

# 사용자가 키를 입력:
#     12345

# s3_download_url = 'https://docs.aws.amazon.com/ko_kr/{12345}/latest/userguide/Welcome.html'


class Openpose_(Resource):
    @api.expect(input_model)
    def post(self):
        result = {
            "code" : 0 ,
            "message" : "요청/연결 오류",
        }
        try:
            # 입력 파싱
            value = request.get_json()
            access_key = str(value.get("access_key"))
            secret_key = str(value.get("secret_key"))
            bucket_name = str(value.get("bucket_name"))
            s3_key = str(value.get("s3_key"))
            local_path = str(value.get("local_path"))
            output_path = str(value.get("output_path"))
        except:
            import traceback
            e = traceback.format_exc()
            result["code"] = 0
            result["message"] = e
            return jsonify(result)
        
        try:
            #버킷 이름, 권한키, 시크릿 키 통해 client 연결
            s3_task = S3_Task(
                access_key=access_key,
                secret_key=secret_key,
                bucket_name=bucket_name
            )
        #     #객체 정보
        #     object_info_list = s3_task.get_object_info_list(s3_prefix)
            
        #     s3_path_list = [x.get('Key') for x in object_info_list.get('Contents')]
            
        #     #다운로드
        #     s3_task.download_s3_object(s3_path_list[1], input_path)
            
        #     #결과값을 output_path 저장
        #     worker(input_path=input_path, output_path=output_path)
            
        #     #output_path를 업로드함
        #     s3_task.upload_s3_object(s3_path_list[1], output_path)
            
        #     result["code"] = 1
        #     result["message"] = "성공"
        # except:
        #     import traceback
        #     e = traceback.format_exc()
        #     result["code"] = -1
        #     result["message"] = e
        #     return jsonify(result)
            s3_task = S3_Task(access_key, secret_key, bucket_name)
            
            # S3에서 파일 다운로드
            s3_task.download_s3_object(s3_key, local_path)

            # 파일을 처리하여 결과 생성
            worker(local_path, output_path, None)

            # 결과를 S3 "pose" 버킷에 업로드
            s3_task.upload_s3_object(output_path, "pose/" + s3_key.split('/')[-1])

            result["code"] = 1
            result["message"] = "성공"
        except Exception as e:
            import traceback
            result["code"] = -1
            result["message"] = traceback.format_exc()
        
        return jsonify(result)

#엔드포인트
api.add_resource(Openpose_, '/path_input')

if __name__ == '__main__':
    app.run(port=8081)
    
    
