import time
import numpy as np
import pandas as pd
import cv2
from collections import deque
from config import *

class StateDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_analyzing = False

        # 상태 지속 시간 기록
        self.state_start_time = time.time()
        self.current_state = 0

        # 상태 변화를 부드럽게 처리하기 위한 버퍼
        self.state_buffer = deque(maxlen=5)
        self.confidence_buffer = deque(maxlen=5)

        # 상태별 임계값 설정
        self.thresholds = {
            'eye_aspect_ratio': 0.21,   # 눈 감김 판단 임계값
            'mouth_aspect_ratio': 0.5,  # 하품 판단 임계값
            'hand_face_distance': 100,  # 손-얼굴 가까움 판단 임계값
            'hand_mouth_distance': 70,  # 생각하는 포즈 판단 임계값
            'hand_cross_distance': 50,  # 손 교차 판단 임계값
            'hand_cross_angle': 30      # 손 교차 각도 판단 임계값 (도 단위)
        }

    def start_analysis(self):
        self.is_analyzing = True
        print("상태 감지 시작")

    def stop_analysis(self):
        self.is_analyzing = False
        print("상태 감지 중지")

    def create_engineered_features(self, face_data, hand_data):
        features = {}

        # None 체크 및 타입 체크 추가
        if face_data is None:
            face_data = {}
        elif not isinstance(face_data, dict):
            # 딕셔너리가 아닌 경우 빈 딕셔너리로 대체
            print(f"Warning: face_data is not a dictionary (type: {type(face_data)})")
            face_data = {}
            
        if hand_data is None:
            hand_data = {}
        elif not isinstance(hand_data, dict):
            # 딕셔너리가 아닌 경우 빈 딕셔너리로 대체
            print(f"Warning: hand_data is not a dictionary (type: {type(hand_data)})")
            hand_data = {}

        # 안전하게 딕셔너리 업데이트
        if isinstance(face_data, dict):
            features.update(face_data)
        if isinstance(hand_data, dict):
            features.update(hand_data)

        # 결측값 처리
        for key in features:
            if features[key] is None:
                features[key] = 0

        # 1. 눈 종횡비 (Eye Aspect Ratio)
        for side in ['left', 'right']:
            eye_top_y = features.get(f'{side}_eye_top_y', 0)
            eye_bottom_y = features.get(f'{side}_eye_bottom_y', 0)
            eye_top_x = features.get(f'{side}_eye_top_x', 0)
            eye_bottom_x = features.get(f'{side}_eye_bottom_x', 0)

            eye_inner_x = features.get(f'{side}_eye_inner_x', 0)
            eye_inner_y = features.get(f'{side}_eye_inner_y', 0)
            eye_outer_x = features.get(f'{side}_eye_outer_x', 0)
            eye_outer_y = features.get(f'{side}_eye_outer_y', 0)

            features[f'{side}_eye_height'] = np.sqrt(
                (eye_top_y - eye_bottom_y)**2 +
                (eye_top_x - eye_bottom_x)**2
            )

            features[f'{side}_eye_width'] = np.sqrt(
                (eye_outer_x - eye_inner_x)**2 +
                (eye_outer_y - eye_inner_y)**2
            )

            if features[f'{side}_eye_width'] > 0:
                features[f'{side}_eye_ratio'] = features[f'{side}_eye_height'] / features[f'{side}_eye_width']
            else:
                features[f'{side}_eye_ratio'] = 0

        left_eye_ratio = features.get('left_eye_ratio', 0)
        right_eye_ratio = features.get('right_eye_ratio', 0)
        features['eye_aspect_ratio'] = (left_eye_ratio + right_eye_ratio) / 2

        # 2. 입 종횡비 (Mouth Aspect Ratio)
        mouth_top_y = features.get('mouth_top_y', 0)
        mouth_bottom_y = features.get('mouth_bottom_y', 0)
        mouth_top_x = features.get('mouth_top_x', 0)
        mouth_bottom_x = features.get('mouth_bottom_x', 0)

        mouth_left_x = features.get('mouth_left_x', 0)
        mouth_left_y = features.get('mouth_left_y', 0)
        mouth_right_x = features.get('mouth_right_x', 0)
        mouth_right_y = features.get('mouth_right_y', 0)

        features['mouth_height'] = np.sqrt(
            (mouth_top_y - mouth_bottom_y)**2 +
            (mouth_top_x - mouth_bottom_x)**2
        )

        features['mouth_width'] = np.sqrt(
            (mouth_right_x - mouth_left_x)**2 +
            (mouth_right_y - mouth_left_y)**2
        )

        if features['mouth_width'] > 0:
            features['mouth_aspect_ratio'] = features['mouth_height'] / features['mouth_width']
        else:
            features['mouth_aspect_ratio'] = 0

        # 3. 손-얼굴 거리
        face_center_x = features.get('face_center_x', 0)
        face_center_y = features.get('face_center_y', 0)

        right_hand_x = features.get('right_hand_x', 0)
        right_hand_y = features.get('right_hand_y', 0)
        right_hand_detected = features.get('right_hand_detected', False)

        if right_hand_detected and right_hand_x != 0:
            features['right_hand_to_face_distance'] = np.sqrt(
                (right_hand_x - face_center_x)**2 +
                (right_hand_y - face_center_y)**2
            )
        else:
            features['right_hand_to_face_distance'] = 1000

        left_hand_x = features.get('left_hand_x', 0)
        left_hand_y = features.get('left_hand_y', 0)
        left_hand_detected = features.get('left_hand_detected', False)

        if left_hand_detected and left_hand_x != 0:
            features['left_hand_to_face_distance'] = np.sqrt(
                (left_hand_x - face_center_x)**2 +
                (left_hand_y - face_center_y)**2
            )
        else:
            features['left_hand_to_face_distance'] = 1000

        features['min_hand_to_face_distance'] = min(
            features['right_hand_to_face_distance'],
            features['left_hand_to_face_distance']
        )

        # 4. 손-입 거리
        mouth_center_x = features.get('mouth_center_x', 0)
        mouth_center_y = features.get('mouth_center_y', 0)

        if right_hand_detected and right_hand_x != 0:
            features['right_hand_to_mouth_distance'] = np.sqrt(
                (right_hand_x - mouth_center_x)**2 +
                (right_hand_y - mouth_center_y)**2
            )
        else:
            features['right_hand_to_mouth_distance'] = 1000

        if left_hand_detected and left_hand_x != 0:
            features['left_hand_to_mouth_distance'] = np.sqrt(
                (left_hand_x - mouth_center_x)**2 +
                (left_hand_y - mouth_center_y)**2
            )
        else:
            features['left_hand_to_mouth_distance'] = 1000

        features['min_hand_to_mouth_distance'] = min(
            features['right_hand_to_mouth_distance'],
            features['left_hand_to_mouth_distance']
        )

        # 5. 얼굴 기울기
        face_top_y = features.get('face_top_y', 0)
        face_bottom_y = features.get('face_bottom_y', 0)
        face_left_x = features.get('face_left_x', 0)
        face_right_x = features.get('face_right_x', 0)

        face_width = abs(face_right_x - face_left_x)
        if face_width > 0:
            features['face_vertical_ratio'] = abs(face_top_y - face_bottom_y) / face_width
        else:
            features['face_vertical_ratio'] = 0

        # 6. 손 교차 감지
        if left_hand_detected and right_hand_detected:
            hand_distance = np.sqrt((left_hand_x - right_hand_x) ** 2 + (left_hand_y - right_hand_y) ** 2)
            features['hand_cross_distance'] = hand_distance

            delta_x = right_hand_x - left_hand_x
            delta_y = right_hand_y - left_hand_y
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            features['hand_cross_angle'] = abs(angle)
        else:
            features['hand_cross_distance'] = 1000
            features['hand_cross_angle'] = 0

        return features

    def rule_based_detection(self, face_data, hand_data):
        features = self.create_engineered_features(face_data, hand_data)

        # 1. 눈 감김 감지 (State 1)
        if features['eye_aspect_ratio'] < self.thresholds['eye_aspect_ratio']:
            confidence = 1.0 - (features['eye_aspect_ratio'] / self.thresholds['eye_aspect_ratio'])
            return 1, min(max(confidence, 0.5), 0.9)

        # 2. 하품 감지 (State 2)
        if features['mouth_aspect_ratio'] > self.thresholds['mouth_aspect_ratio']:
            confidence = min((features['mouth_aspect_ratio'] / self.thresholds['mouth_aspect_ratio'] - 1.0) * 2.0, 1.0)
            return 2, min(max(confidence, 0.5), 0.9)

        # 3. 생각하는 포즈 감지 (State 3)
        if features['min_hand_to_mouth_distance'] < self.thresholds['hand_mouth_distance']:
            confidence = 1.0 - (features['min_hand_to_mouth_distance'] / self.thresholds['hand_mouth_distance'])
            return 3, min(max(confidence, 0.5), 0.9)

        # 4. 눈 비빔 감지 (State 4)
        if features['min_hand_to_face_distance'] < self.thresholds['hand_face_distance']:
            confidence = 1.0 - (features['min_hand_to_face_distance'] / self.thresholds['hand_face_distance'])
            return 4, min(max(confidence, 0.5), 0.9)

        # 5. 손 교차 감지 (State 5)
        if features['hand_cross_distance'] < self.thresholds['hand_cross_distance'] and \
          abs(features['hand_cross_angle'] - 90) < self.thresholds['hand_cross_angle']:
           confidence = 1.0 - (features['hand_cross_distance'] / self.thresholds['hand_cross_distance'])
           return 5, min(max(confidence, 0.5), 0.9)
           
        # 기본 상태 (State 0)
        return 0, 1.0

    def analyze_state(self, frame, face_data, hand_data):
        """
        프레임, 얼굴 및 손 데이터를 기반으로 상태를 분석합니다.
        
        Parameters:
        - frame: 원본 영상 프레임
        - face_data: 얼굴 특징점 데이터
        - hand_data: 손 특징점 데이터
        
        Returns:
        - processed_frame: 처리된 영상 프레임
        - final_state: 검출된 상태 (0-5)
        - state_info: 상태 정보를 담은 딕셔너리 {'confidence': 신뢰도, 'duration': 지속 시간}
        """
        try:
            # face_data가 numpy 배열인 경우 처리
            if isinstance(face_data, np.ndarray):
                print(f"Converting face_data from ndarray to dictionary, shape: {face_data.shape}")
                face_dict = {}
                face_data = face_dict
            
            # hand_data가 numpy 배열인 경우 처리
            if isinstance(hand_data, np.ndarray):
                print(f"Converting hand_data from ndarray to dictionary, shape: {hand_data.shape}")
                hand_dict = {}
                hand_data = hand_dict
                
            state, confidence = self.rule_based_detection(face_data, hand_data)
            
            # 상태 버퍼에 추가
            self.state_buffer.append(state)
            self.confidence_buffer.append(confidence)
            
            # 가장 빈번한 상태 결정
            state_counts = {}
            for s, c in zip(self.state_buffer, self.confidence_buffer):
                if s not in state_counts:
                    state_counts[s] = 0
                state_counts[s] += c
            
            # 가장 높은 신뢰도의 상태 선택
            max_confidence = 0
            final_state = 0
            
            for s, total_conf in state_counts.items():
                if total_conf > max_confidence:
                    max_confidence = total_conf
                    final_state = s
            
            # 상태 변화 감지
            if final_state != self.current_state:
                self.current_state = final_state
                self.state_start_time = time.time()
            
            # 상태 지속 시간 계산
            duration = time.time() - self.state_start_time
            
            # 상태 정보를 딕셔너리로 반환
            state_info = {
                'confidence': confidence,
                'duration': duration
            }
            
            # 프레임에 상태 정보 표시 (옵션)
            processed_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            if self.is_analyzing and final_state > 0:
                # 상태에 따라 프레임에 텍스트/시각화 추가 (옵션)
                cv2.putText(processed_frame, f"State: {final_state} ({STATE_KOREAN.get(final_state, '알 수 없음')})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Duration: {duration:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return processed_frame, final_state, state_info
        
        except Exception as e:
            print(f"상태 분석 중 오류 발생: {e}")
            print(f"face_data 타입: {type(face_data)}")
            print(f"hand_data 타입: {type(hand_data)}")
            # 오류 발생 시 기본값 반환
            default_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            return default_frame, 0, {'confidence': 1.0, 'duration': 0.0}