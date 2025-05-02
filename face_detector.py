"""
Face detection and analysis module.
Handles face detection, landmark extraction, and facial metrics calculations.
"""
import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 중요한 랜드마크 인덱스 정의
        self.landmarks_indices = {
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 291, 0, 17, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13]
        }
        
        # 특정 랜드마크 인덱스
        self.special_landmarks = {
            'left_eye_inner': 133,
            'left_eye_outer': 173,
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_top': 13,  
            'mouth_bottom': 14,
            'face_top': 10,
            'face_bottom': 152,
            'face_left': 234,
            'face_right': 454
        }
        
    def detect_face(self, frame):
        """얼굴 감지 및 랜드마크 추출"""
        if frame is None:
            return False, None, {}
            
        height, width = frame.shape[:2]
        
        # BGR에서 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 랜드마크 감지
        results = self.face_mesh.process(rgb_frame)
        
        # 결과 시각화 및 데이터 추출
        face_data = {'face_detected': False}
        
        # 얼굴이 감지되었는지 확인
        if not results.multi_face_landmarks:
            return False, frame, face_data
            
        face_data['face_detected'] = True
        
        # 첫 번째 얼굴만 처리
        face_landmarks = results.multi_face_landmarks[0]
        
        # 랜드마크 그리기
        annotated_frame = frame.copy()
        
        # 얼굴 윤곽선 그리기
        for landmark_type, indices in self.landmarks_indices.items():
            color = (0, 255, 0)  # 기본 색상 (녹색)
            thickness = 1
            
            if landmark_type == 'left_eye' or landmark_type == 'right_eye':
                color = (255, 0, 0)  # 눈은 파란색
                thickness = 1
            elif landmark_type == 'mouth':
                color = (0, 0, 255)  # 입은 빨간색
                thickness = 1
                
            # 연결선 그리기
            points = []
            for idx in indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * width), int(lm.y * height)
                points.append((x, y))
                
            # 다각형 그리기
            cv2.polylines(annotated_frame, [np.array(points)], True, color, thickness)
        
        # 특정 랜드마크 추출 및 데이터에 저장
        for name, idx in self.special_landmarks.items():
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * width), int(lm.y * height)
            
            # 특정 랜드마크에 점 표시
            cv2.circle(annotated_frame, (x, y), 2, (0, 255, 255), -1)
            
            # 데이터에 좌표 저장
            face_data[f'{name}_x'] = x
            face_data[f'{name}_y'] = y
            
        # 얼굴 중심점 계산
        face_center_x = int((face_data['face_left_x'] + face_data['face_right_x']) / 2)
        face_center_y = int((face_data['face_top_y'] + face_data['face_bottom_y']) / 2)
        face_data['face_center_x'] = face_center_x
        face_data['face_center_y'] = face_center_y
        
        # 입 중심점 계산
        mouth_center_x = int((face_data['mouth_left_x'] + face_data['mouth_right_x']) / 2)
        mouth_center_y = int((face_data['mouth_top_y'] + face_data['mouth_bottom_y']) / 2)
        face_data['mouth_center_x'] = mouth_center_x
        face_data['mouth_center_y'] = mouth_center_y
        
        # 얼굴 중심에 큰 점 표시
        cv2.circle(annotated_frame, (face_center_x, face_center_y), 5, (255, 255, 0), -1)
        
        # 얼굴 검출 영역에 사각형 표시
        x_min = face_data['face_left_x']
        y_min = face_data['face_top_y']
        x_max = face_data['face_right_x']
        y_max = face_data['face_bottom_y']
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 얼굴 상태 표시 영역
        status_bg_height = 30
        cv2.rectangle(annotated_frame, (x_min, y_min - status_bg_height), (x_max, y_min), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (x_min, y_min - status_bg_height), (x_max, y_min), (0, 255, 0), 1)
        
        # 얼굴 상태 텍스트
        cv2.putText(
            annotated_frame, 
            "얼굴 감지됨", 
            (x_min + 10, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return True, annotated_frame, face_data
        
    def close(self):
        """리소스 해제"""
        self.face_mesh.close()