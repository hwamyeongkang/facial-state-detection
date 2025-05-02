"""
Model training module.
Handles training, evaluating, and saving the facial state detection model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import *
from config import MODEL_FILE, SCALER_FILE

# SMOTE 임포트 시도 (설치되지 않았을 경우 대비)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("경고: SMOTE를 사용할 수 없습니다. pip install imbalanced-learn으로 설치하세요.")

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_model(self):
        """저장된 모델과 스케일러 로드"""
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
                try:
                    self.model = pickle.load(open(MODEL_FILE, 'rb'))
                    self.scaler = pickle.load(open(SCALER_FILE, 'rb'))
                    print("모델과 스케일러를 성공적으로 로드했습니다.")
                    return True
                except Exception as e:
                    print(f"모델 로드 중 오류 발생: {e}")
                    return False
            else:
                print("모델 또는 스케일러 파일이 존재하지 않습니다.")
                return False
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            return False

    def train_model(self):
        """모델 학습 및 저장"""
        try:
            # 데이터 디렉토리 및 파일 확인
            if not os.path.exists(CSV_FILE):
                print(f"CSV 파일이 존재하지 않습니다: {CSV_FILE}")
                print(f"현재 작업 디렉토리: {os.getcwd()}")
                print(f"CSV_FILE 절대 경로: {os.path.abspath(CSV_FILE)}")
                return False, 0.0
                
            # 데이터 로드
            try:
                df = pd.read_csv(CSV_FILE, encoding='utf-8')
                print(f"CSV 파일 로드 성공: {CSV_FILE}")
                print(f"로드된 데이터 크기: {len(df)} 행, {len(df.columns)} 열")
            except Exception as e:
                print(f"CSV 파일 로드 실패: {e}")
                return False, 0.0
            
            if len(df) == 0:
                print("CSV 파일이 비어 있습니다.")
                return False, 0.0
                
            print(f"데이터 로드 완료: {len(df)} 샘플")
            
            # 'state' 열의 고유 값 확인
            print("'state' 열의 고유 값:")
            unique_states = df['state'].unique()
            print(unique_states)
            
            # 'state' 열에서 숫자가 아닌 값 처리
            non_numeric_mask = pd.to_numeric(df['state'], errors='coerce').isna()
            problem_count = non_numeric_mask.sum()
            
            if problem_count > 0:
                print(f"경고: 'state' 열에 숫자가 아닌 값이 {problem_count}개 있습니다. 이 데이터는 제외됩니다.")
                problem_values = df.loc[non_numeric_mask, 'state'].unique()
                print(f"문제가 있는 상태 값: {problem_values}")
                
                # 문제 데이터 제거
                print("문제 데이터 제거 중...")
                df = df[~non_numeric_mask]
                print(f"제거 후 데이터 크기: {len(df)} 행")
                
                if len(df) == 0:
                    print("오류: 모든 데이터가 제거되었습니다. 유효한 데이터가 없습니다.")
                    return False, 0.0
            
            # timestamp 컬럼 명시적 제거
            if 'timestamp' in df.columns:
                print("timestamp 컬럼 제거")
                df = df.drop(columns=['timestamp'])
            
            # 데이터 형태 확인
            print("데이터 컬럼:", df.columns.tolist())
            
            # 필수 컬럼 확인
            if 'state' not in df.columns:
                print("오류: 'state' 컬럼이 데이터에 없습니다.")
                return False, 0.0
            
            # 클래스 확인
            state_counts = df['state'].value_counts()
            print(f"클래스 분포: {state_counts}")
            
            if len(state_counts) < 2:
                print("오류: 분류를 위해선 최소 2개 이상의 클래스가 필요합니다.")
                return False, 0.0
            
            # 데이터 타입 확인 및 변환 (문자열 -> 숫자)
            numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in numeric_cols:
                if col != 'state':  # state 컬럼은 나중에 처리
                    try:
                        # 문자열 타입을 숫자로 변환 시도
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"컬럼 '{col}'을(를) 숫자 타입으로 변환했습니다.")
                    except Exception as e:
                        print(f"컬럼 '{col}' 변환 중 오류: {e}")
            
            # 부울 컬럼 명시적 변환
            bool_columns = ['right_hand_detected', 'left_hand_detected', 'face_detected']
            for col in bool_columns:
                if col in df.columns:
                    # 문자열 'True'/'False'를 부울로 변환
                    df[col] = df[col].astype(str).map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
                    # 결측값 처리
                    df[col] = df[col].fillna(0)
                    print(f"컬럼 '{col}'을(를) 부울(0/1)로 변환했습니다.")
            
            # 결측치 처리 전 데이터 타입 확인
            print("변환 후 데이터 타입:")
            print(df.dtypes)
            
            # 특성 엔지니어링 적용
            df = self.create_engineered_features(df)
            
            # 결측값 확인 및 처리
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print("결측값이 있는 컬럼:", null_counts[null_counts > 0].to_dict())
                print("결측값을 0으로 대체합니다.")
                df = df.fillna(0)
            
            # 레이블과 특성 분리 - 안전하게 정수로 변환
            try:
                y = df['state'].astype(int)  # 상태를 정수로 변환
                print(f"변환된 상태 값: {y.unique()}")
            except Exception as e:
                print(f"상태 값 변환 중 오류: {e}")
                print("CSV 파일을 확인하고 수정 후 다시 시도하세요.")
                return False, 0.0
            
            # 시간 관련 특성과 상태 레이블은 제외
            drop_cols = ['state']
            X = df.drop(columns=drop_cols, errors='ignore')
            
            # 데이터 타입 최종 확인
            print("학습 데이터 형태:", X.shape)
            
            # 모든 컬럼이 숫자 타입인지 확인
            non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"경고: 숫자가 아닌 컬럼이 있습니다: {non_numeric_cols}")
                print("이 컬럼을 제거합니다.")
                X = X.drop(columns=non_numeric_cols)
                print("제거 후 학습 데이터 형태:", X.shape)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"모델 정확도: {accuracy:.2f}")
            print(classification_report(y_test, y_pred))

            # 모델 저장 경로 확인
            print(f"모델 파일 경로: {MODEL_FILE}")
            print(f"스케일러 파일 경로: {SCALER_FILE}")
            
            # 모델 저장 전 경로 확인 및 생성
            model_dir = os.path.dirname(MODEL_FILE)
            print(f"모델 디렉토리 경로: {model_dir}")
            print(f"모델 디렉토리 존재 여부: {os.path.exists(model_dir)}")
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"모델 저장 디렉토리를 생성했습니다: {model_dir}")

            try:
                with open(MODEL_FILE, 'wb') as f_model, open(SCALER_FILE, 'wb') as f_scaler:
                    pickle.dump(self.model, f_model)
                    pickle.dump(self.scaler, f_scaler)
                print(f"모델과 스케일러 저장 완료: {MODEL_FILE}, {SCALER_FILE}")
            except Exception as e:
                print(f"모델 저장 중 오류: {e}")
                import traceback
                traceback.print_exc()

            return True, accuracy

        except Exception as e:
            print(f"모델 학습 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def create_engineered_features(self, data):
        """특성 엔지니어링을 통해 새로운 특성을 생성"""
        try:
            df = data.copy()
            
            # 모든 데이터가 숫자 타입인지 확인
            for col in df.columns:
                if col == 'state':
                    continue
                    
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"특성 엔지니어링: 컬럼 '{col}'을(를) 숫자 타입으로 변환했습니다.")
                    except Exception as e:
                        print(f"특성 엔지니어링: 컬럼 '{col}' 변환 실패, 제거됩니다: {e}")
                        df = df.drop(columns=[col])
            
            # 1. 눈 종횡비 (Eye Aspect Ratio) - 눈 감김 탐지에 유용
            for side in ['left', 'right']:
                # 필요한 컬럼이 있는지 확인
                required_cols = [
                    f'{side}_eye_top_y', f'{side}_eye_bottom_y',
                    f'{side}_eye_top_x', f'{side}_eye_bottom_x',
                    f'{side}_eye_outer_x', f'{side}_eye_inner_x',
                    f'{side}_eye_outer_y', f'{side}_eye_inner_y'
                ]
                
                if all(col in df.columns for col in required_cols):
                    # 눈의 세로 거리 (세로 랜드마크 거리의 평균)
                    df[f'{side}_eye_height'] = np.sqrt(
                        (df[f'{side}_eye_top_y'] - df[f'{side}_eye_bottom_y'])**2 +
                        (df[f'{side}_eye_top_x'] - df[f'{side}_eye_bottom_x'])**2
                    )
                    
                    # 눈의 가로 거리
                    df[f'{side}_eye_width'] = np.sqrt(
                        (df[f'{side}_eye_outer_x'] - df[f'{side}_eye_inner_x'])**2 +
                        (df[f'{side}_eye_outer_y'] - df[f'{side}_eye_inner_y'])**2
                    )
                    
                    # 눈 종횡비 (EAR) - 0으로 나누기 방지
                    df[f'{side}_eye_ratio'] = np.where(
                        df[f'{side}_eye_width'] > 0,
                        df[f'{side}_eye_height'] / df[f'{side}_eye_width'],
                        0
                    )
                else:
                    print(f"경고: {side} 눈 관련 열이 누락되었습니다. 기본값 0 사용.")
                    df[f'{side}_eye_height'] = 0
                    df[f'{side}_eye_width'] = 0
                    df[f'{side}_eye_ratio'] = 0
            
            # 눈 종횡비의 평균 - 양쪽 눈의 감김 정도
            if 'left_eye_ratio' in df.columns and 'right_eye_ratio' in df.columns:
                df['eye_aspect_ratio'] = (df['left_eye_ratio'] + df['right_eye_ratio']) / 2
            else:
                df['eye_aspect_ratio'] = 0
                print("경고: 눈 종횡비 계산에 필요한 열이 누락되었습니다. 기본값 0 사용.")
            
            # 2. 입 종횡비 (Mouth Aspect Ratio) - 하품/입 벌림 탐지에 유용
            mouth_cols = [
                'mouth_top_y', 'mouth_bottom_y', 'mouth_top_x', 'mouth_bottom_x',
                'mouth_right_x', 'mouth_left_x', 'mouth_right_y', 'mouth_left_y'
            ]
            
            if all(col in df.columns for col in mouth_cols):
                df['mouth_height'] = np.sqrt(
                    (df['mouth_top_y'] - df['mouth_bottom_y'])**2 +
                    (df['mouth_top_x'] - df['mouth_bottom_x'])**2
                )
                
                df['mouth_width'] = np.sqrt(
                    (df['mouth_right_x'] - df['mouth_left_x'])**2 +
                    (df['mouth_right_y'] - df['mouth_left_y'])**2
                )
                
                # 입 종횡비 - 0으로 나누기 방지
                df['mouth_aspect_ratio'] = np.where(
                    df['mouth_width'] > 0,
                    df['mouth_height'] / df['mouth_width'],
                    0
                )
            else:
                print("경고: 입 관련 열이 누락되었습니다. 기본값 0 사용.")
                df['mouth_height'] = 0
                df['mouth_width'] = 0
                df['mouth_aspect_ratio'] = 0
            
            # 3. 손-얼굴 거리 (손으로 눈 비빔 탐지에 유용)
            if 'face_center_x' in df.columns and 'face_center_y' in df.columns:
                face_center_x = df['face_center_x']
                face_center_y = df['face_center_y']
                
                # 오른손 거리
                if 'right_hand_x' in df.columns and 'right_hand_y' in df.columns:
                    # 부울 조건 안전하게 처리
                    if 'right_hand_detected' in df.columns:
                        right_hand_detected = df['right_hand_detected'] > 0  # 0/1로 변환된 값 사용
                    else:
                        right_hand_detected = pd.Series(True, index=df.index)
                    
                    right_hand_x = df['right_hand_x']
                    right_hand_y = df['right_hand_y']
                    
                    # 안전하게 계산
                    df['right_hand_to_face_distance'] = np.where(
                        right_hand_detected & (right_hand_x != 0),
                        np.sqrt((right_hand_x - face_center_x)**2 + (right_hand_y - face_center_y)**2),
                        1000
                    )
                else:
                    df['right_hand_to_face_distance'] = 1000
                    print("경고: 오른손 관련 열이 누락되었습니다. 기본값 1000 사용.")
                    
                # 왼손 거리
                if 'left_hand_x' in df.columns and 'left_hand_y' in df.columns:
                    # 부울 조건 안전하게 처리
                    if 'left_hand_detected' in df.columns:
                        left_hand_detected = df['left_hand_detected'] > 0  # 0/1로 변환된 값 사용
                    else:
                        left_hand_detected = pd.Series(True, index=df.index)
                    
                    left_hand_x = df['left_hand_x']
                    left_hand_y = df['left_hand_y']
                    
                    # 안전하게 계산
                    df['left_hand_to_face_distance'] = np.where(
                        left_hand_detected & (left_hand_x != 0),
                        np.sqrt((left_hand_x - face_center_x)**2 + (left_hand_y - face_center_y)**2),
                        1000
                    )
                else:
                    df['left_hand_to_face_distance'] = 1000
                    print("경고: 왼손 관련 열이 누락되었습니다. 기본값 1000 사용.")
                    
                # 손-얼굴 거리의 최소값 (더 가까운 손만 고려)
                if 'right_hand_to_face_distance' in df.columns and 'left_hand_to_face_distance' in df.columns:
                    df['min_hand_to_face_distance'] = df[['right_hand_to_face_distance', 'left_hand_to_face_distance']].min(axis=1)
                else:
                    df['min_hand_to_face_distance'] = 1000
            else:
                print("경고: 얼굴 중심 열이 누락되었습니다. 손-얼굴 거리에 기본값 사용.")
                df['right_hand_to_face_distance'] = 1000
                df['left_hand_to_face_distance'] = 1000
                df['min_hand_to_face_distance'] = 1000
            
            # 4. 손-입 거리 (생각하는 포즈 탐지에 유용) 
            if 'mouth_center_x' in df.columns and 'mouth_center_y' in df.columns:
                mouth_center_x = df['mouth_center_x']
                mouth_center_y = df['mouth_center_y']
                
                # 오른손-입 거리
                if 'right_hand_x' in df.columns and 'right_hand_y' in df.columns:
                    # 부울 조건 안전하게 처리
                    if 'right_hand_detected' in df.columns:
                        right_hand_detected = df['right_hand_detected'] > 0  # 0/1로 변환된 값 사용
                    else:
                        right_hand_detected = pd.Series(True, index=df.index)
                    
                    right_hand_x = df['right_hand_x']
                    right_hand_y = df['right_hand_y']
                    
                    # 안전하게 계산
                    df['right_hand_to_mouth_distance'] = np.where(
                        right_hand_detected & (right_hand_x != 0),
                        np.sqrt((right_hand_x - mouth_center_x)**2 + (right_hand_y - mouth_center_y)**2),
                        1000
                    )
                else:
                    df['right_hand_to_mouth_distance'] = 1000
                    
                # 왼손-입 거리
                if 'left_hand_x' in df.columns and 'left_hand_y' in df.columns:
                    # 부울 조건 안전하게 처리
                    if 'left_hand_detected' in df.columns:
                        left_hand_detected = df['left_hand_detected'] > 0  # 0/1로 변환된 값 사용
                    else:
                        left_hand_detected = pd.Series(True, index=df.index)
                    
                    left_hand_x = df['left_hand_x']
                    left_hand_y = df['left_hand_y']
                    
                    # 안전하게 계산
                    df['left_hand_to_mouth_distance'] = np.where(
                        left_hand_detected & (left_hand_x != 0),
                        np.sqrt((left_hand_x - mouth_center_x)**2 + (left_hand_y - mouth_center_y)**2),
                        1000
                    )
                else:
                    df['left_hand_to_mouth_distance'] = 1000
                    
                # 손-입 거리의 최소값
                if 'right_hand_to_mouth_distance' in df.columns and 'left_hand_to_mouth_distance' in df.columns:
                    df['min_hand_to_mouth_distance'] = df[['right_hand_to_mouth_distance', 'left_hand_to_mouth_distance']].min(axis=1)
                else:
                    df['min_hand_to_mouth_distance'] = 1000
            else:
                print("경고: 입 중심 열이 누락되었습니다. 손-입 거리에 기본값 사용.")
                df['right_hand_to_mouth_distance'] = 1000
                df['left_hand_to_mouth_distance'] = 1000
                df['min_hand_to_mouth_distance'] = 1000
            
            # 5. 얼굴 기울기 (고개 숙임 탐지에 유용)
            face_cols = ['face_top_y', 'face_bottom_y', 'face_right_x', 'face_left_x']
            
            if all(col in df.columns for col in face_cols):
                # 0으로 나누기 방지
                df['face_vertical_ratio'] = np.where(
                    np.abs(df['face_right_x'] - df['face_left_x']) > 0,
                    np.abs(df['face_top_y'] - df['face_bottom_y']) / np.abs(df['face_right_x'] - df['face_left_x']),
                    0
                )
            else:
                print("경고: 얼굴 윤곽 열이 누락되었습니다. 얼굴 기울기에 기본값 0 사용.")
                df['face_vertical_ratio'] = 0
                
            return df
            
        except Exception as e:
            print(f"특성 엔지니어링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return data  # 원본 데이터 반환