
"""
Data collection module.
Handles collecting, storing, and managing facial and hand landmark data.
"""
import os
import time
import csv
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from config import *

class DataCollector:
    def __init__(self):
        self.is_collecting = False
        self.current_state = 0  # 초기 상태: 정상
        self.collection_count = 0
        
        # config.py에서 MAX_SAMPLES_PER_STATE 변수 확인
        try:
            self.max_samples = MAX_SAMPLES_PER_STATE
            print(f"최대 샘플 수 설정: {self.max_samples}")
        except NameError:
            print("경고: config.py에 MAX_SAMPLES_PER_STATE가 정의되지 않았습니다. 기본값 100 사용.")
            self.max_samples = 100
        
        self.last_save_time = 0  # 마지막 저장 시간
        self.save_interval = 0.1  # 데이터 저장 간격 (초)
        
        # 필요한 디렉토리 생성
        self.ensure_directories()
        
        # CSV 헤더 생성 (파일이 없는 경우)
        self.check_create_csv()
        
        # 데이터 균형을 모니터링하기 위한 상태별 카운터
        self.state_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.load_existing_counts()
        
        # 저장할 때 문제가 있었는지 확인하기 위한 플래그
        self.save_error_reported = False
        self.fallback_csv_file = os.path.join(os.getcwd(), "facial_data_fallback.csv")
    
    def ensure_directories(self):
        """필요한 디렉토리가 존재하는지 확인하고 없으면 생성"""
        try:
            # 데이터 폴더 생성
            os.makedirs(DATA_FOLDER, exist_ok=True)
            print(f"데이터 폴더 확인: {DATA_FOLDER}")
            
            # 스크린샷 폴더 생성
            os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
            print(f"스크린샷 폴더 확인: {SCREENSHOT_FOLDER}")
            
            # 모델 폴더 생성 (필요한 경우)
            model_dir = os.path.dirname(MODEL_FILE)
            if model_dir and model_dir != "":
                os.makedirs(model_dir, exist_ok=True)
                print(f"모델 폴더 확인: {model_dir}")
            
            # CSV 파일 디렉토리 확인
            csv_dir = os.path.dirname(CSV_FILE)
            if csv_dir and csv_dir != "":
                os.makedirs(csv_dir, exist_ok=True)
                print(f"CSV 파일 폴더 확인: {csv_dir}")
                
            # 디렉토리 쓰기 권한 확인
            if os.access(DATA_FOLDER, os.W_OK):
                print(f"데이터 폴더에 쓰기 권한이 있습니다.")
            else:
                print(f"경고: 데이터 폴더에 쓰기 권한이 없습니다!")
                # 대체 경로 시도
                alt_dir = os.getcwd()
                if os.access(alt_dir, os.W_OK):
                    print(f"대체로 현재 작업 디렉토리를 사용합니다: {alt_dir}")
                else:
                    print("심각한 권한 문제: 현재 작업 디렉토리에도 쓰기 권한이 없습니다!")
                
        except Exception as e:
            print(f"디렉토리 생성 중 오류: {e}")
            print("기본 경로를 사용합니다.")
    
    def get_valid_csv_path(self):
        """유효한 CSV 파일 경로 반환"""
        # 먼저 기본 경로 시도
        if os.path.exists(os.path.dirname(CSV_FILE)) and os.access(os.path.dirname(CSV_FILE), os.W_OK):
            return CSV_FILE
            
        # 기본 경로에 문제가 있으면 대체 경로 사용
        fallback_path = self.fallback_csv_file
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        return fallback_path
    
    def load_existing_counts(self):
        """기존 CSV 파일이 있다면 상태별 샘플 수를 로드"""
        try:
            csv_path = self.get_valid_csv_path()
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if 'state' in df.columns:
                        states = df['state'].value_counts().to_dict()
                        for state, count in states.items():
                            self.state_counts[int(state)] = count
                        print(f"기존 데이터 상태별 분포: {self.state_counts}")
                    else:
                        print("CSV 파일에 'state' 열이 없습니다.")
                except Exception as e:
                    print(f"CSV 파일 읽기 오류: {e}")
        except Exception as e:
            print(f"기존 데이터 로드 중 오류: {e}")
    
    def check_create_csv(self):
        """CSV 파일 존재 확인 및 헤더 생성"""
        try:
            # 유효한 CSV 경로 가져오기
            csv_path = self.get_valid_csv_path()
            print(f"사용할 CSV 파일 경로: {csv_path}")
            
            # CSV 파일이 없거나 빈 경우 헤더 생성
            create_header = False
            
            if not os.path.exists(csv_path):
                create_header = True
                print(f"CSV 파일이 존재하지 않아 새로 생성합니다: {csv_path}")
            elif os.path.getsize(csv_path) == 0:
                create_header = True
                print(f"CSV 파일이 비어 있어 헤더를 추가합니다: {csv_path}")
            
            if create_header:
                try:
                    # 폴더 확인
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # 헤더 생성
                        header = ['timestamp', 'state']
                        
                        # 얼굴 랜드마크 특성
                        face_features = [
                            'face_detected', 'face_center_x', 'face_center_y',
                            'face_top_x', 'face_top_y', 'face_bottom_x', 'face_bottom_y',
                            'face_left_x', 'face_left_y', 'face_right_x', 'face_right_y',
                            'left_eye_inner_x', 'left_eye_inner_y', 'left_eye_outer_x', 'left_eye_outer_y',
                            'left_eye_top_x', 'left_eye_top_y', 'left_eye_bottom_x', 'left_eye_bottom_y',
                            'right_eye_inner_x', 'right_eye_inner_y', 'right_eye_outer_x', 'right_eye_outer_y',
                            'right_eye_top_x', 'right_eye_top_y', 'right_eye_bottom_x', 'right_eye_bottom_y',
                            'mouth_left_x', 'mouth_left_y', 'mouth_right_x', 'mouth_right_y',
                            'mouth_top_x', 'mouth_top_y', 'mouth_bottom_x', 'mouth_bottom_y',
                            'mouth_center_x', 'mouth_center_y'
                        ]
                        
                        # 손 랜드마크 특성
                        hand_features = [
                            'left_hand_detected', 'left_hand_x', 'left_hand_y',
                            'right_hand_detected', 'right_hand_x', 'right_hand_y'
                        ]
                        
                        # 지속 시간 특성
                        duration_features = [
                            'yawn_duration', 'thinking_duration'
                        ]
                        
                        header.extend(face_features)
                        header.extend(hand_features)
                        header.extend(duration_features)
                        
                        writer.writerow(header)
                        print(f"CSV 파일 헤더 생성 성공: {csv_path}")
                        
                        # 파일 생성 확인
                        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                            print("CSV 파일이 성공적으로 생성되었고 헤더가 작성되었습니다.")
                        else:
                            print("경고: CSV 파일 생성은 성공했으나 내용이 비어 있습니다.")
                            
                except Exception as e:
                    print(f"CSV 헤더 생성 중 오류: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"CSV 파일 확인/생성 중 오류: {e}")
            traceback.print_exc()
    
    def start_collection(self, state_code):
        """데이터 수집 시작"""
        self.is_collecting = True
        self.current_state = state_code
        self.collection_count = 0
        self.save_error_reported = False  # 오류 플래그 초기화
        print(f"상태 {state_code}({STATE_KOREAN[state_code]}) 데이터 수집 시작")
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.is_collecting = False
        print(f"상태 {self.current_state}({STATE_KOREAN[self.current_state]}) 데이터 수집 중지, {self.collection_count}개 샘플 수집됨")
        self.collection_count = 0
    
    def set_max_samples(self, max_samples):
        """최대 샘플 수 설정"""
        self.max_samples = max_samples
        print(f"최대 샘플 수 설정: {max_samples}")
    
    def save_to_csv(self, landmarks_data):
        """랜드마크 데이터를 CSV에 저장"""
        if not self.is_collecting:
            return
            
        current_time = time.time()
        
        # 저장 간격 확인 (너무 빈번한 저장 방지)
        if current_time - self.last_save_time < self.save_interval:
            return
            
        self.last_save_time = current_time
        
        # 최대 샘플 수 확인
        if self.collection_count >= self.max_samples:
            self.stop_collection()
            return
            
        try:
            # 타임스탬프 추가
            landmarks_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # 상태 추가
            landmarks_data["state"] = self.current_state
            
            # 결측값 처리
            for key in landmarks_data:
                if landmarks_data[key] is None:
                    landmarks_data[key] = 0
            
            # 사용할 CSV 파일 경로 결정
            csv_path = self.get_valid_csv_path()
            
            # CSV에 저장 (방법 1: csv 모듈 사용)
            success = False
            try:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=landmarks_data.keys())
                    writer.writerow(landmarks_data)
                success = True
            except Exception as e:
                print(f"CSV 저장 방법 1 실패: {e}")
                
                # 방법 2: pandas 사용
                try:
                    df = pd.DataFrame([landmarks_data])
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                    success = True
                except Exception as e2:
                    print(f"CSV 저장 방법 2 실패: {e2}")
                    
                    # 방법 3: 기본 파일 쓰기
                    if not self.save_error_reported:  # 한 번만 경고 출력
                        print("두 가지 저장 방법이 모두 실패했습니다. 대체 로직을 사용합니다.")
                        self.save_error_reported = True
                    
                    try:
                        # 대체 로직: 텍스트 파일에 저장
                        backup_file = os.path.join(os.getcwd(), "facial_data_backup.txt")
                        with open(backup_file, 'a') as f:
                            f.write(f"{landmarks_data['timestamp']},{landmarks_data['state']}\n")
                        success = True  # 텍스트로라도 저장되면 성공으로 간주
                    except Exception as e3:
                        print(f"모든 저장 방법 실패: {e3}")
            
            # 저장 성공 시 카운트 증가
            if success:
                self.collection_count += 1
                self.state_counts[self.current_state] += 1
                
                # 로그 출력 (10개 단위로)
                if self.collection_count % 10 == 0:
                    print(f"상태 {self.current_state}({STATE_KOREAN[self.current_state]}) 데이터 {self.collection_count}/{self.max_samples} 수집됨")
                    
                # 첫 번째 데이터가 저장되면 성공 메시지 출력
                if self.collection_count == 1:
                    print(f"첫 번째 데이터 저장 성공! 경로: {csv_path}")
            else:
                if not self.save_error_reported:
                    print("심각한 오류: 모든 데이터 저장 방법이 실패했습니다.")
                    self.save_error_reported = True
                
        except Exception as e:
            print(f"CSV 저장 처리 중 예기치 않은 오류: {e}")
            traceback.print_exc()
    
    def capture_screenshot(self, frame):
        """현재 프레임을 스크린샷으로 저장"""
        try:
            if frame is None:
                return None
                
            # 현재 시간을 파일명으로 사용
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"state_{self.current_state}_{timestamp}.jpg"
            
            # 스크린샷 폴더 확인 및 생성
            try:
                os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
                filepath = os.path.join(SCREENSHOT_FOLDER, filename)
            except Exception:
                # 스크린샷 폴더에 문제가 있으면 현재 디렉토리 사용
                filepath = os.path.join(os.getcwd(), filename)
            
            # 이미지 저장
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"스크린샷 저장됨: {filepath}")
                return filename
            else:
                print(f"경고: 스크린샷 저장 실패")
                
                # 대체 경로 시도
                alt_filepath = os.path.join(os.getcwd(), filename)
                if filepath != alt_filepath:  # 이미 대체 경로를 사용한 경우는 제외
                    alt_success = cv2.imwrite(alt_filepath, frame)
                    if alt_success:
                        print(f"대체 경로에 스크린샷 저장됨: {alt_filepath}")
                        return filename
                
                return None
        except Exception as e:
            print(f"스크린샷 저장 중 오류: {e}")
            return None
    
    def get_data_balance_info(self):
        """데이터 균형 정보 반환"""
        total = sum(self.state_counts.values())
        if total == 0:
            return "데이터 없음"
            
        balance_info = []
        for state, count in self.state_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            balance_info.append(f"{STATE_KOREAN[state]}: {count}개 ({percentage:.1f}%)")
            
        return ", ".join(balance_info)
    
    def get_data_recommendations(self):
        """데이터 수집 권장사항 반환"""
        if sum(self.state_counts.values()) == 0:
            return "모든 상태의 데이터를 수집하세요."
            
        # 가장 적은 데이터를 가진 상태 찾기
        min_state = min(self.state_counts.items(), key=lambda x: x[1])
        
        # 데이터가 없는 상태 찾기
        empty_states = [state for state, count in self.state_counts.items() if count == 0]
        
        recommendations = []
        
        if empty_states:
            empty_states_text = ", ".join([STATE_KOREAN[s] for s in empty_states])
            recommendations.append(f"데이터가 없는 상태({empty_states_text})의 데이터를 수집하세요.")
        
        if min_state[1] > 0:
            recommendations.append(f"가장 적은 데이터를 가진 상태는 '{STATE_KOREAN[min_state[0]]}'({min_state[1]}개)입니다.")
        
        # 불균형이 심한 경우
        max_state = max(self.state_counts.items(), key=lambda x: x[1])
        if max_state[1] > 0 and min_state[1] > 0 and max_state[1] / min_state[1] > 3:
            recommendations.append(f"데이터 불균형이 심합니다. '{STATE_KOREAN[min_state[0]]}' 상태의 데이터를 더 수집하세요.")
        
        if not recommendations:
            recommendations.append("데이터 균형이 적절합니다.")
            
        return " ".join(recommendations) 