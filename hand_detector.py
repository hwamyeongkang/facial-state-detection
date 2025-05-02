"""
Hand detection and analysis module.
Handles hand detection, landmark extraction, and hand-face interaction metrics.
"""

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        try:
            # MediaPipe Hands 초기화
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # 주요 랜드마크 인덱스
            self.landmarks = {
                'WRIST': 0,          # 손목
                'THUMB_TIP': 4,      # 엄지 끝
                'INDEX_FINGER_TIP': 8,  # 검지 끝
                'MIDDLE_FINGER_TIP': 12, # 중지 끝
                'RING_FINGER_TIP': 16,   # 약지 끝
                'PINKY_TIP': 20      # 소지 끝
            }
            print("HandDetector 초기화 성공")
        except Exception as e:
            print(f"HandDetector 초기화 중 오류: {e}")
            raise
        
    def detect_hands(self, frame):
        """손 감지 및 랜드마크 추출"""
        try:
            if frame is None:
                print("입력 프레임이 None입니다.")
                return False, None, {}
                
            height, width = frame.shape[:2]
            
            # 결과 시각화 및 데이터 초기화
            hand_data = {
                'left_hand_detected': False,
                'right_hand_detected': False
            }
            
            # 결과 프레임 초기화
            annotated_frame = frame.copy()
            
            # BGR에서 RGB로 변환
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"색상 변환 중 오류: {e}")
                return False, annotated_frame, hand_data
            
            # 랜드마크 감지
            try:
                results = self.hands.process(rgb_frame)
            except Exception as e:
                print(f"손 랜드마크 처리 중 오류: {e}")
                return False, annotated_frame, hand_data
            
            # 손이 감지되었는지 확인
            if not results.multi_hand_landmarks:
                return False, annotated_frame, hand_data
                
            # 감지된 모든 손 처리
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # 손 타입 확인 (왼손/오른손)
                    if hand_idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[hand_idx]
                        hand_type = handedness.classification[0].label  # 'Left' 또는 'Right'
                        
                        # 손 랜드마크 그리기
                        try:
                            self.mp_drawing.draw_landmarks(
                                annotated_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                        except Exception as e:
                            print(f"랜드마크 그리기 중 오류: {e}")
                        
                        # 손목 위치 가져오기
                        try:
                            wrist = hand_landmarks.landmark[self.landmarks['WRIST']]
                            wrist_x = int(wrist.x * width)
                            wrist_y = int(wrist.y * height)
                        except Exception as e:
                            print(f"손목 랜드마크 처리 중 오류: {e}")
                            wrist_x, wrist_y = 0, 0
                        
                        # 손가락 끝 위치 계산
                        finger_tips = {}
                        for name, idx in self.landmarks.items():
                            try:
                                if name != 'WRIST':
                                    lm = hand_landmarks.landmark[idx]
                                    x, y = int(lm.x * width), int(lm.y * height)
                                    finger_tips[name] = (x, y)
                                    
                                    # 손가락 끝에 점 표시
                                    cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
                            except Exception as e:
                                print(f"{name} 랜드마크 처리 중 오류: {e}")
                        
                        # 손의 중심점 계산
                        try:
                            tip_xs = [pos[0] for pos in finger_tips.values()]
                            tip_ys = [pos[1] for pos in finger_tips.values()]
                            
                            if tip_xs and tip_ys:  # 빈 리스트가 아닌 경우에만
                                hand_center_x = int(sum(tip_xs) / len(tip_xs))
                                hand_center_y = int(sum(tip_ys) / len(tip_ys))
                            else:
                                hand_center_x = wrist_x
                                hand_center_y = wrist_y
                        except Exception as e:
                            print(f"손 중심점 계산 중 오류: {e}")
                            hand_center_x, hand_center_y = wrist_x, wrist_y
                        
                        # 손 타입에 따른 데이터 저장
                        try:
                            if hand_type.lower() == 'left':
                                hand_data['left_hand_detected'] = True
                                hand_data['left_hand_x'] = hand_center_x
                                hand_data['left_hand_y'] = hand_center_y
                                
                                # 손 상태 표시
                                self.draw_hand_status(annotated_frame, wrist_x, wrist_y, "왼손", (0, 255, 255))
                            else:  # 'Right'
                                hand_data['right_hand_detected'] = True
                                hand_data['right_hand_x'] = hand_center_x
                                hand_data['right_hand_y'] = hand_center_y
                                
                                # 손 상태 표시
                                self.draw_hand_status(annotated_frame, wrist_x, wrist_y, "오른손", (255, 0, 255))
                        except Exception as e:
                            print(f"손 데이터 저장 중 오류: {e}")
                except Exception as e:
                    print(f"손 처리 중 오류: {e}")
            
            # 적어도 하나의 손이 감지되었는지 확인
            hand_detected = hand_data['left_hand_detected'] or hand_data['right_hand_detected']
            
            return hand_detected, annotated_frame, hand_data
            
        except Exception as e:
            print(f"손 감지 중 일반 오류: {e}")
            # 최소한의 응답 반환
            if 'annotated_frame' in locals():
                return False, annotated_frame, {'left_hand_detected': False, 'right_hand_detected': False}
            elif 'frame' in locals() and frame is not None:
                return False, frame.copy(), {'left_hand_detected': False, 'right_hand_detected': False}
            else:
                return False, None, {'left_hand_detected': False, 'right_hand_detected': False}
    
    def draw_hand_status(self, frame, x, y, text, color):
        """손 상태 텍스트 표시"""
        try:
            if frame is None:
                return
                
            # 좌표 범위 확인
            h, w = frame.shape[:2]
            if x < 0 or x >= w or y < 0 or y >= h:
                print(f"텍스트 좌표가 프레임 범위를 벗어남: ({x}, {y}), 프레임 크기: {w}x{h}")
                x = min(max(0, x), w-1)
                y = min(max(0, y), h-1)
            
            # 배경 사각형
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 사각형 좌표 계산 및 범위 검사
            rect_x1 = max(0, x - 5)
            rect_y1 = max(0, y - text_size[1] - 10)
            rect_x2 = min(w-1, x + text_size[0] + 5)
            rect_y2 = min(h-1, y)
            
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            
            # 텍스트
            text_x = max(0, x)
            text_y = max(text_size[1], y - 5)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            print(f"손 상태 표시 중 오류: {e}")
        
    def detect_hand_gesture(self, hand_landmarks, width, height):
        """손 제스처 감지 (예: 주먹, 손가락 가리키기 등)"""
        try:
            if not hand_landmarks:
                return "알 수 없음"
                
            # 손가락 끝과 중간 관절의 y 좌표 비교로 손가락이 접혔는지 확인
            fingers_up = []
            
            try:
                # 엄지는 x 좌표로 판단 (수평 방향)
                thumb_tip = hand_landmarks.landmark[self.landmarks['THUMB_TIP']]
                thumb_mcp = hand_landmarks.landmark[2]  # 엄지 중수골 관절
                
                # 엄지가 펴졌는지 확인 (x 좌표 비교)
                fingers_up.append(thumb_tip.x > thumb_mcp.x)
            except Exception as e:
                print(f"엄지 제스처 감지 중 오류: {e}")
                fingers_up.append(False)
            
            # 다른 손가락은 y 좌표로 판단 (수직 방향)
            for finger_name in ['INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']:
                try:
                    finger_idx = self.landmarks[finger_name]
                    finger_tip = hand_landmarks.landmark[finger_idx]
                    finger_pip = hand_landmarks.landmark[finger_idx - 2]  # PIP 관절 (끝에서 2번째 관절)
                    
                    # 손가락 끝이 PIP 관절보다 위에 있으면 손가락이 펴진 것
                    fingers_up.append(finger_tip.y < finger_pip.y)
                except Exception as e:
                    print(f"{finger_name} 제스처 감지 중 오류: {e}")
                    fingers_up.append(False)
            
            # 제스처 판단
            if all(fingers_up):
                return "손바닥"
            elif not any(fingers_up):
                return "주먹"
            elif len(fingers_up) > 1 and fingers_up[1] and not any(fingers_up[2:]):
                return "검지 가리키기"
            elif len(fingers_up) > 2 and fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]):
                return "V 사인"
            else:
                return "기타 제스처"
                
        except Exception as e:
            print(f"제스처 감지 중 오류: {e}")
            return "알 수 없음"
            
    def close(self):
        """리소스 해제"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
                print("HandDetector 리소스 해제 성공")
        except Exception as e:
            print(f"HandDetector 리소스 해제 중 오류: {e}")