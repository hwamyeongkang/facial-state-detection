"""
Configuration file for facial state detection system.
Contains constants, thresholds, and color settings.
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 상태 감지 조건 설정
BORED_THRESHOLD = 0.8        # 얼굴이 20% 이상 작아지면 지루함 (0.8배 이하)
YAWN_THRESHOLD = 0.05        # 입이 5cm 이상 벌어진 것으로 간주
YAWN_DURATION = 3.0          # 하품 감지 유지 시간 (초)
THINKING_DURATION = 5.0      # 고민 중 상태 유지 시간 (초)
EYEBROW_THRESHOLD = 0.9      # 미간 거리가 10% 이상 짧아지면 불만족 (0.9배 이하)
MOUTH_UP_THRESHOLD = 0.01    # 입 높이가 1cm 이상 올라가면 불만족

# 상태 코드 정의
STATE_LABELS = {
    0: 'satisfied',    # 만족 (기본 상태)
    1: 'bored',        # 지루함
    2: 'tired',        # 피곤함
    3: 'thinking',     # 고민 중
    4: 'dissatisfied'  # 불만족
}

# 한글 상태 이름
STATE_KOREAN = {
    0: '만족(기본)',
    1: '지루함',
    2: '피곤함',
    3: '고민 중',
    4: '불만족'
}

# 상태별 색상 (BGR 형식)
STATE_COLORS = {
    0: (46, 204, 113),   # 만족: 녹색
    1: (52, 152, 219),   # 지루함: 파란색
    2: (0, 0, 255),      # 피곤함: 빨간색
    3: (255, 165, 0),    # 고민 중: 주황색
    4: (142, 68, 173)    # 불만족: 보라색
}

# GUI 색상 (RGB 형식)
GUI_COLORS = {
    'bg_dark': '#2c3e50',      # 배경색 (어두운)
    'bg_mid': '#34495e',       # 배경색 (중간)
    'text': '#ecf0f1',         # 텍스트 기본색
    'highlight': '#3498db',    # 강조색
    'green': '#2ecc71',        # 녹색 (버튼 등)
    'red': '#e74c3c',          # 빨간색 (경고 등)
    'orange': '#e67e22',       # 주황색 (주의 등)
    'yellow': '#f1c40f',       # 노란색
    'purple': '#9b59b6'        # 보라색
}

# 현재 작업 디렉토리를 기준으로 절대 경로 설정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 작업 디렉토리

# 데이터 파일 경로
DATA_FOLDER = os.path.join(ROOT_DIR, "facial_data")
MODEL_FOLDER = os.path.join(ROOT_DIR, "models")
SCREENSHOT_FOLDER = os.path.join(ROOT_DIR, "screenshots")
CSV_FILE = os.path.join(DATA_FOLDER, "facial_landmarks.csv")
MODEL_FILE = os.path.join(MODEL_FOLDER, "state_detector_model.pkl")
SCALER_FILE = os.path.join(MODEL_FOLDER, "state_detector_scaler.pkl")  # 추가: 스케일러 파일 경로

# 기본 샘플 설정
MAX_SAMPLES_PER_STATE = 100  # 각 상태별 최대 샘플 수 (DEFAULT_MAX_SAMPLES에서 변경)

# 필요한 디렉토리 생성 함수
def ensure_directories():
    """필요한 모든 디렉토리가 존재하는지 확인하고 없으면 생성"""
    directories = [DATA_FOLDER, MODEL_FOLDER, SCREENSHOT_FOLDER]
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"디렉토리 생성됨: {directory}")
            else:
                print(f"디렉토리 확인됨: {directory}")
        except Exception as e:
            print(f"오류: 디렉토리 생성 중 문제 발생: {e}")
            raise  # 예외를 다시 발생시켜 프로그램 종료

# 경로 설정 정보 출력 (디버깅용)
def print_path_info():
    """경로 설정 정보 출력"""
    print("\n=== 설정 정보 ===")
    print(f"작업 디렉토리: {ROOT_DIR}")
    print(f"데이터 폴더: {DATA_FOLDER} (존재 여부: {os.path.exists(DATA_FOLDER)})")
    print(f"모델 폴더: {MODEL_FOLDER} (존재 여부: {os.path.exists(MODEL_FOLDER)})")
    print(f"스크린샷 폴더: {SCREENSHOT_FOLDER} (존재 여부: {os.path.exists(SCREENSHOT_FOLDER)})")
    print(f"CSV 파일: {CSV_FILE} (존재 여부: {os.path.exists(CSV_FILE)})")
    # config.py에서 다음과 같이 수정
    MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'facial_state_model.pkl')
    SCALER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'facial_state_scaler.pkl')
    print(f"최대 샘플 수: {MAX_SAMPLES_PER_STATE}")
    print("=================\n")

# 초기화 시 디렉토리 확인 및 생성
ensure_directories()
print_path_info()
ensure_directories()