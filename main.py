import os
import sys
import time
import threading
import subprocess
import cv2
import traceback

# 프로젝트 내 모듈 임포트
try:
    from config import *
    from face_detector import FaceDetector
    from hand_detector import HandDetector
    from state_detector import StateDetector
    from data_collector import DataCollector
    from model_trainer import ModelTrainer
    from gui import FacialStateGUI
    from music_player import MusicPlayer  # 음악 플레이어 모듈 추가
except ImportError as e:
    print(f"모듈 가져오기 오류: {e}")
    print("모든 필요한 모듈이 올바르게 설치되었는지 확인하세요.")
    sys.exit(1)

class FacialStateDetectionApp:
    def __init__(self):
        try:
            print("FacialStateDetectionApp 초기화 중...")
            
            # GUI 초기화
            try:
                self.gui = FacialStateGUI()
                self.register_event_handlers()
                print("GUI 초기화 성공")
            except Exception as e:
                print(f"GUI 초기화 중 오류: {e}")
                raise
            
            # 탐지기 초기화
            try:
                self.face_detector = FaceDetector()
                print("얼굴 탐지기 초기화 성공")
            except Exception as e:
                print(f"얼굴 탐지기 초기화 중 오류: {e}")
                self.gui.show_message_box("오류", "얼굴 탐지기 초기화 실패. MediaPipe가 올바르게 설치되었는지 확인하세요.", is_error=True)
                raise
                
            try:
                self.hand_detector = HandDetector()
                print("손 탐지기 초기화 성공")
            except Exception as e:
                print(f"손 탐지기 초기화 중 오류: {e}")
                self.gui.show_message_box("오류", "손 탐지기 초기화 실패. MediaPipe가 올바르게 설치되었는지 확인하세요.", is_error=True)
                raise
                
            try:
                self.state_detector = StateDetector()
                print("상태 탐지기 초기화 성공")
            except Exception as e:
                print(f"상태 탐지기 초기화 중 오류: {e}")
                self.gui.show_message_box("오류", "상태 탐지기 초기화 실패.", is_error=True)
                raise
            
            # 음악 플레이어 초기화 추가
            try:
                self.music_player = MusicPlayer(self.gui)
                print("음악 플레이어 초기화 성공")
            except Exception as e:
                print(f"음악 플레이어 초기화 중 오류: {e}")
                self.gui.show_message_box("오류", "음악 플레이어 초기화 실패. Selenium이 올바르게 설치되었는지 확인하세요.", is_error=True)
            
            # 상태-음악 매핑 설정
            self.state_music_map = {
                1: "알파파",       # 눈 감김 (피곤함) -> 알파파 음악
                2: "감마파",       # 하품 (지루함) -> 감마파 음악
                3: "모짜르트",     # 생각하는 포즈 (심사숙고) -> 모짜르트
                4: "명상음악",     # 눈 비빔 (피로감) -> 명상음악
                5: "에너지음악"    # 손 교차 (불만족) -> 에너지 음악
            }
            
            # 상태 지속 임계값 설정 (초 단위)
            self.state_duration_threshold = 5.0  # 5초 이상 지속되면 음악 추천
            
            # 음악 자동 추천 설정
            self.auto_recommend_music = False  # 기본적으로 비활성화
            self.last_music_state = 0  # 마지막으로 음악이 재생된 상태
            
            # 데이터 및 모델 관련 클래스 초기화
            try:
                self.data_collector = DataCollector()
                self.model_trainer = ModelTrainer()
                print("데이터/모델 관련 클래스 초기화 성공")
            except Exception as e:
                print(f"데이터/모델 관련 클래스 초기화 중 오류: {e}")
                self.gui.show_message_box("오류", "데이터 수집 또는 모델 학습 초기화 실패.", is_error=True)
                raise
            
            # 웹캠 초기화
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("웹캠을 열 수 없습니다.")
                    self.webcam_available = False
                    self.gui.show_webcam_error()
                else:
                    self.webcam_available = True
                    print("웹캠 초기화 성공")
            except Exception as e:
                print(f"웹캠 초기화 중 오류: {e}")
                self.webcam_available = False
                self.gui.show_webcam_error()
            
            # 상태 변수 초기화
            self.is_running = False
            self.current_frame = None
            self.processed_frame = None
            self.video_playing = False
            self.video_process = None
            
            # 모델 로드 시도
            try:
                model_loaded = self.model_trainer.load_model()
                if model_loaded:
                    self.state_detector.model = self.model_trainer.model
                    self.state_detector.scaler = self.model_trainer.scaler
                    self.gui.update_model_status(True)
                    print("모델 로드 성공")
                else:
                    self.gui.update_model_status(False)
                    print("모델 로드 실패 (파일이 없거나 오류 발생)")
            except Exception as e:
                print(f"모델 로드 중 오류: {e}")
                self.gui.update_model_status(False)
            
            print("FacialStateDetectionApp 초기화 완료")
            
        except Exception as e:
            print(f"애플리케이션 초기화 중 심각한 오류: {e}")
            traceback.print_exc()
            if hasattr(self, 'gui'):
                self.gui.show_message_box("심각한 오류", f"애플리케이션 초기화 중 오류가 발생했습니다: {e}", is_error=True)
            sys.exit(1)

    def register_event_handlers(self):
        """GUI 이벤트 핸들러 등록"""
        try:
            self.gui.on_start_stop = self.toggle_start_stop
            self.gui.on_collection_toggle = self.toggle_collection
            self.gui.on_analysis_toggle = self.toggle_analysis
            self.gui.on_train_model = self.start_training
            self.gui.on_capture_screenshot = self.capture_screenshot
            self.gui.on_state_selected = self.on_state_selected
            self.gui.on_max_samples_selected = self.on_max_samples_selected
            self.gui.on_open_data_folder = self.open_data_folder
            self.gui.on_closing = self.on_closing
            self.gui.on_auto_music_toggle = self.toggle_auto_music  # 음악 자동 추천 토글 핸들러 추가
            print("GUI 이벤트 핸들러 등록 완료")
        except Exception as e:
            print(f"이벤트 핸들러 등록 중 오류: {e}")
            raise

    def toggle_start_stop(self):
        """시작/중지 버튼 토글 처리"""
        try:
            if self.is_running:
                self.is_running = False
                self.gui.set_start_stop_button_state(False)
                self.stop_video()
                self.gui.update_status("중지됨")
                print("애플리케이션 중지됨")
            else:
                if not self.webcam_available:
                    self.gui.update_status("웹캠을 사용할 수 없습니다.", warning=True)
                    print("웹캠을 사용할 수 없어 시작할 수 없습니다.")
                    return
                
                self.is_running = True
                self.gui.set_start_stop_button_state(True)
                self.play_video()
                self.gui.update_status("실행 중...")
                self.update_webcam()
                print("애플리케이션 시작됨")
                
                if not self.state_detector.model:
                    self.gui.update_status("모델이 없습니다. 먼저 모델을 학습하거나 로드하세요.", warning=True)
                    print("경고: 모델이 로드되지 않았습니다.")
        except Exception as e:
            print(f"시작/중지 토글 중 오류: {e}")
            self.gui.update_status(f"오류: {e}", warning=True)

    def play_video(self):
        """배경 비디오 재생"""
        try:
            gamma_path = os.path.join(os.getcwd(), "gamma.mp4")
            if os.path.exists(gamma_path):
                self.video_process = subprocess.Popen(
                    ['start', '', gamma_path], shell=True
                )
                self.video_playing = True
                print("🎬 gamma.mp4 재생 시작")
            else:
                print("gamma.mp4 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"비디오 재생 오류: {e}")

    def stop_video(self):
        """배경 비디오 중지"""
        try:
            if self.video_process and self.video_playing:
                self.video_process.terminate()
                self.video_playing = False
                print("🎬 gamma.mp4 재생 중단")
        except Exception as e:
            print(f"비디오 종료 오류: {e}")

    def toggle_collection(self):
        """데이터 수집 토글 처리"""
        try:
            if self.data_collector.is_collecting:
                self.data_collector.stop_collection()
                self.gui.set_collection_button_state(False)
                self.gui.update_status("데이터 수집 중지됨")
                print("데이터 수집 중지")
            else:
                state_code = int(self.gui.state_combo.get()[0])
                self.data_collector.start_collection(state_code)
                self.gui.set_collection_button_state(True)
                self.gui.update_sample_count(0)
                self.gui.update_status(f"{STATE_KOREAN[state_code]} 상태 데이터 수집 시작")
                print(f"{STATE_KOREAN[state_code]} 상태 데이터 수집 시작")
        except Exception as e:
            print(f"데이터 수집 토글 중 오류: {e}")
            self.gui.update_status(f"데이터 수집 중 오류: {e}", warning=True)

    def toggle_analysis(self):
        """상태 감지 토글 처리"""
        try:
            if self.state_detector.is_analyzing:
                self.state_detector.stop_analysis()
                self.gui.set_analysis_button_state(False)
                self.gui.update_status("상태 감지 중지됨")
                print("상태 감지 중지")
            else:
                self.state_detector.start_analysis()
                self.gui.set_analysis_button_state(True)
                self.gui.update_status("상태 감지 시작됨")
                print("상태 감지 시작")
        except Exception as e:
            print(f"상태 감지 토글 중 오류: {e}")
            self.gui.update_status(f"상태 감지 중 오류: {e}", warning=True)

    def toggle_auto_music(self):
        """음악 자동 추천 토글"""
        try:
            self.auto_recommend_music = not self.auto_recommend_music
            status = "활성화" if self.auto_recommend_music else "비활성화"
            self.gui.update_status(f"음악 자동 추천 {status}됨")
            self.gui.set_auto_music_button_state(self.auto_recommend_music)
            print(f"음악 자동 추천 {status}됨")
            
            # 비활성화 시 브라우저 닫기
            if not self.auto_recommend_music and hasattr(self, 'music_player'):
                self.music_player.close_browser()
                self.last_music_state = 0
        except Exception as e:
            print(f"음악 자동 추천 토글 중 오류): {e}")

    def start_training(self):
        """모델 학습 시작"""
        try:
            if os.path.exists(CSV_FILE):
                self.gui.set_training_button_state(True)
                self.gui.update_status("모델 학습 시작...")
                print("모델 학습 시작...")
                threading.Thread(target=self.train_model_thread).start()
            else:
                self.gui.update_status("학습 데이터가 없습니다.", warning=True)
                print("학습 데이터가 없습니다.")
        except Exception as e:
            print(f"학습 시작 중 오류: {e}")
            self.gui.update_status(f"학습 시작 중 오류: {e}", warning=True)

    def train_model_thread(self):
        """모델 학습 스레드"""
        try:
            success, accuracy = self.model_trainer.train_model()
            if success:
                self.state_detector.model = self.model_trainer.model
                self.state_detector.scaler = self.model_trainer.scaler
                self.gui.root.after(0, lambda: self.gui.update_status(f"모델 학습 완료! 정확도: {accuracy:.2f}"))
                self.gui.root.after(0, lambda: self.gui.update_model_status(True))
                print(f"모델 학습 완료! 정확도: {accuracy:.2f}")
            else:
                self.gui.root.after(0, lambda: self.gui.update_status("모델 학습 실패!", warning=True))
                print("모델 학습 실패!")
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            self.gui.root.after(0, lambda: self.gui.update_status(f"모델 학습 중 오류: {e}", warning=True))
        finally:
            self.gui.root.after(0, lambda: self.gui.set_training_button_state(False))

    def recommend_music_for_state(self, state, duration):
        """상태에 따라 음악 추천"""
        # 상태가 0이거나 설정된 임계값보다 지속 시간이 짧으면 무시
        if state == 0 or duration < self.state_duration_threshold:
            return
            
        # 이미 같은 상태에 대한 음악이 재생 중이면 무시
        if state == self.last_music_state and self.music_player.music_playing:
            return
            
        # 상태에 해당하는 음악 검색어 가져오기
        music_search = self.state_music_map.get(state)
        if not music_search:
            return
            
        # 상태별 메시지 구성
        state_messages = {
            1: "눈 감김 감지 - 휴식이 필요합니다. 알파파 음악을 재생합니다.",
            2: "하품 감지 - 활력이 필요합니다. 감마파 음악을 재생합니다.",
            3: "생각하는 포즈 감지 - 집중력 향상을 위한 모짜르트를 재생합니다.",
            4: "눈 비빔 감지 - 눈의 피로감 완화를 위한 명상음악을 재생합니다.",
            5: "불만족 감지 - 긍정적 에너지를 위한 음악을 재생합니다."
        }
        
        message = state_messages.get(state, f"상태 {state} 감지 - 음악을 재생합니다.")
        self.gui.update_status(message)
        print(message)
        
        # 음악 재생
        success = self.music_player.play_music(music_search)
        if success:
            self.last_music_state = state

    def capture_screenshot(self):
        """현재 프레임 스크린샷 저장"""
        try:
            if self.processed_frame is not None:
                filename = self.data_collector.capture_screenshot(self.processed_frame)
                if filename:
                    self.gui.update_status(f"스크린샷 저장됨: {filename}")
                    print(f"스크린샷 저장됨: {filename}")
                else:
                    self.gui.update_status("스크린샷 저장 실패", warning=True)
                    print("스크린샷 저장 실패")
            else:
                self.gui.update_status("캡처할 화면이 없습니다.", warning=True)
                print("캡처할 화면이 없습니다.")
        except Exception as e:
            print(f"스크린샷 캡처 중 오류: {e}")
            self.gui.update_status(f"스크린샷 캡처 중 오류: {e}", warning=True)

    def on_state_selected(self, state_code):
        """상태 선택 처리"""
        try:
            self.data_collector.current_state = state_code
            print(f"상태 선택됨: {state_code} ({STATE_KOREAN[state_code]})")
        except Exception as e:
            print(f"상태 선택 중 오류: {e}")

    def on_max_samples_selected(self, max_samples):
        """최대 샘플 수 선택 처리"""
        try:
            self.data_collector.set_max_samples(max_samples)
            print(f"최대 샘플 수 설정: {max_samples}")
        except Exception as e:
            print(f"최대 샘플 수 설정 중 오류: {e}")

    def open_data_folder(self):
        """데이터 폴더 열기"""
        try:
            abs_path = os.path.abspath(DATA_FOLDER)
            if os.path.exists(abs_path):
                if os.name == 'nt':
                    os.startfile(abs_path)
                else:
                    import subprocess
                    subprocess.run(['xdg-open', abs_path])
                self.gui.update_status(f"데이터 폴더 열기: {abs_path}")
                print(f"데이터 폴더 열기: {abs_path}")
            else:
                self.gui.update_status("데이터 폴더가 존재하지 않습니다.", warning=True)
                print("데이터 폴더가 존재하지 않습니다.")
        except Exception as e:
            print(f"폴더 열기 오류: {e}")
            self.gui.update_status(f"폴더 열기 오류: {e}", warning=True)

    def update_webcam(self):
        """웹캠 프레임 업데이트 및 처리"""
        # 전체 메서드를 try-except로 감싸서 어떤 예외도 처리
        try:
            # 앱이 실행 중이 아니거나 웹캠을 사용할 수 없으면 종료
            if not self.is_running or not self.webcam_available:
                return
                
            # 웹캠에서 프레임 읽기
            ret, self.current_frame = self.cap.read()
            if not ret:
                self.gui.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                self.is_running = False
                self.gui.set_start_stop_button_state(False)
                return
                
            # 좌우 반전 (거울 효과)
            self.current_frame = cv2.flip(self.current_frame, 1)

            # 얼굴 감지
            face_detected = False
            frame_with_face = self.current_frame.copy()
            face_data = {}
            
            try:
                face_detected, frame_with_face, face_data = self.face_detector.detect_face(self.current_frame)
            except Exception as e:
                print(f"얼굴 감지 중 오류: {e}")
            
            self.gui.update_face_status(face_detected)

            # 손 감지
            hand_detected = False
            frame_with_face_and_hand = frame_with_face
            hand_data = {}
            
            try:
                hand_detected, frame_with_face_and_hand, hand_data = self.hand_detector.detect_hands(frame_with_face)
            except Exception as e:
                print(f"손 감지 중 오류: {e}")
            
            self.gui.update_hand_status(hand_detected)

            # 상태 분석
            detected_state = 0
            state_info = {'duration': 0.0, 'confidence': 0}
            
            try:
                self.processed_frame, detected_state, state_info = self.state_detector.analyze_state(
                    frame_with_face_and_hand, face_data, hand_data)
            except Exception as e:
                print(f"상태 분석 중 오류: {e}")
                self.processed_frame = frame_with_face_and_hand
            
            self.gui.update_detected_state(detected_state, duration=state_info.get('duration', 0.0))
            self.gui.update_confidence(state_info.get('confidence', 0))
            
            # 음악 자동 추천이 활성화되어 있고 충분한 신뢰도를 가진 경우
            if self.auto_recommend_music and state_info.get('confidence', 0) > 0.6:
                self.recommend_music_for_state(
                    detected_state, 
                    state_info.get('duration', 0.0)
                )

            # GUI 업데이트
            try:
                self.gui.update_webcam_display(self.processed_frame)
            except Exception as e:
                print(f"GUI 업데이트 중 오류: {e}")

            # 데이터 수집
            if self.data_collector.is_collecting:
                try:
                    landmarks_data = {}
                    landmarks_data.update(face_data)
                    landmarks_data.update(hand_data)
                    landmarks_data["yawn_duration"] = state_info.get('duration', 0) if detected_state == 2 else 0
                    landmarks_data["thinking_duration"] = state_info.get('duration', 0) if detected_state == 3 else 0

                    self.data_collector.save_to_csv(landmarks_data)
                    self.gui.update_sample_count(self.data_collector.collection_count)
                except Exception as e:
                    print(f"데이터 수집 중 오류: {e}")

            # 다음 프레임 예약
            self.gui.root.after(10, self.update_webcam)
            
        except Exception as e:
            print(f"웹캠 업데이트 중 일반 오류: {e}")
            traceback.print_exc()
            self.gui.update_status(f"웹캠 업데이트 중 오류: {e}", warning=True)
            # 오류가 발생해도 계속 실행될 수 있도록 다음 프레임 예약
            self.gui.root.after(100, self.update_webcam)

    def on_closing(self):
        """프로그램 종료 처리"""
        try:
            self.is_running = False
            
            # 비디오 중지
            if self.video_playing:
                self.stop_video()
                
            # 웹캠 해제
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
                
            # 탐지기 리소스 해제
            if hasattr(self, 'face_detector'):
                self.face_detector.close()
                
            if hasattr(self, 'hand_detector'):
                self.hand_detector.close()
            
            # 음악 플레이어 브라우저 종료 추가
            if hasattr(self, 'music_player'):
                self.music_player.close_browser()
                
            print("프로그램 종료")
        except Exception as e:
            print(f"프로그램 종료 중 오류: {e}")

    def run(self):
        """메인 애플리케이션 실행"""
        try:
            self.gui.run()
        except Exception as e:
            print(f"애플리케이션 실행 중 오류: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    try:
        print("표정 모델학습 애플리케이션 시작...")
        app = FacialStateDetectionApp()
        app.run()
    except Exception as e:
        print(f"치명적인 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)        