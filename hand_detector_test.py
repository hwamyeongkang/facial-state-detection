import cv2
import time
from hand_detector import HandDetector

def test_hand_detector():
    print("=== 손 감지기 테스트 시작 ===")
    
    # 카메라 초기화
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        print("웹캠 초기화 성공")
    except Exception as e:
        print(f"카메라 초기화 중 오류: {e}")
        return
    
    # HandDetector 초기화
    try:
        detector = HandDetector()
        print("HandDetector 초기화 성공")
    except Exception as e:
        print(f"HandDetector 초기화 중 오류: {e}")
        cap.release()
        return
    
    # FPS 측정용 변수
    prev_time = 0
    curr_time = 0
    
    # 메인 루프
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 프레임 좌우 반전 (거울 효과)
            frame = cv2.flip(frame, 1)
            
            # 손 감지
            try:
                hand_detected, frame_with_hands, hand_data = detector.detect_hands(frame)
                
                # 손 감지 결과 표시
                status_text = "손 감지됨" if hand_detected else "손 감지되지 않음"
                cv2.putText(frame_with_hands, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 손 데이터 표시
                if hand_detected:
                    y_offset = 60
                    for key, value in hand_data.items():
                        if key.endswith('_detected') and value:
                            hand_type = key.split('_')[0]
                            x_val = hand_data.get(f'{hand_type}_hand_x', 0)
                            y_val = hand_data.get(f'{hand_type}_hand_y', 0)
                            text = f"{hand_type} 손: ({x_val}, {y_val})"
                            cv2.putText(frame_with_hands, text, (10, y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_offset += 30
                
                # FPS 계산 및 표시
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                cv2.putText(frame_with_hands, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 결과 표시
                cv2.imshow("Hand Detection", frame_with_hands)
                
            except Exception as e:
                print(f"손 감지 중 오류: {e}")
                cv2.imshow("Hand Detection", frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자가 'q'를 눌러 종료합니다.")
                break
                
    except Exception as e:
        print(f"메인 루프 중 오류: {e}")
    finally:
        # 리소스 해제
        try:
            cap.release()
            detector.close()
            cv2.destroyAllWindows()
            print("리소스 해제 완료")
        except Exception as e:
            print(f"리소스 해제 중 오류: {e}")
    
    print("=== 손 감지기 테스트 종료 ===")

if __name__ == "__main__":
    test_hand_detector()