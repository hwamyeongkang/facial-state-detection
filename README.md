## 🧠 Webcam Learning Behavior Monitor

**실시간 웹컴을 통해 학습자의 지중 상태 및 학습 태도를 감지하는 AI 기반 배열 통계 앱**

![preview](./screenshot.png)

---

### 📌 주요 기능 (Features)

* 👁️ 안면 및 손 랜드링 추적 (MediaPipe, InsightFace)
* 🧠 상태 감지: 조음, 하품, 눈비빘, 손짓기, 손그릇가락 (불만족) 등
* 🧪 머신롤링 기반 감지 모델 학습 및 적용
* 📸 1초 간격 자동 스크린샷 캐프쳐
* 🔒 Firebase 기반 회원가입 및 로그인
* 🖥️ Tkinter / PyQt 기능 GUI 제공
* 🌐 Flask 서버 연동 가능
* 🔎 RAG 기능 사용자 질의응답 도입 가능 (선택)

---

### 💠 사용 기술 (Tech Stack)

| Layer        | Tech                                         |
| ------------ | -------------------------------------------- |
| **Frontend** | Tkinter, PyQt5, HTML (Flask templates)       |
| **Backend**  | Flask, Firebase, InsightFace, MediaPipe      |
| **AI/ML**    | Scikit-learn, Pandas, OpenCV, RAG (optional) |
| **Infra**    | Python 3.9+, VSCode, Firebase                |

---

### 🚀 실행 방법 (How to Run)

1. **필수 패키지 설치**

   ```bash
   pip install -r requirements.txt
   ```

2. **Firebase 설정**

   * `firebase_key.json` 파일을 프로젝트 루트에 배치
   * `.env` 또는 코드 내 `FIREBASE_API_KEY` 설정

3. **실행**

   ```bash
   python main.py
   ```

---

### 📂 디렉토리 구조 (Structure)

```
📁 project/
│
├── main.py                     # 실행 진입점
├── face_detector.py            # 안면 감지 로직
├── state_detector.py           # 상태 분석 로직
├── model_trainer.py            # ML 학습 및 저장
├── firebase_auth.py            # 로그인/회원가입 로직
├── templates/                  # Flask HTML 템플릿
└── firebase_key.json           # Firebase 서비스 계정 키
```

---

### 🧪 상태 코드 정의 (State Labels)

| 코드 | 상태         |
| -- | ---------- |
| 0  | 기본 / 지중    |
| 1  | 눈 감기       |
| 2  | 하품         |
| 3  | 생각 중       |
| 4  | 눈 비빘       |
| 5  | 불만족 (손 교차) |

---

### 📸 캐프쳐 예시
![image](https://github.com/user-attachments/assets/f00a801e-49f7-4c47-96d1-390ddc509de2)


---

### 👨‍💼 기억 방법 (Contributing)

PR 및 이슈 등록 환영합니다. 🤍
