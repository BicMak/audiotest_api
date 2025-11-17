# 실시간 음성 인식 시스템 (Raw PCM Streaming)

## 📁 폴더 구조
```
project/
├── app.py              # 통신서버 (8000 포트)
├── judge_server.py     # 판단서버 (9000 포트)
├── static/
│   ├── streaming.html  # 프론트엔드
│   └── processor.js    # 오디오 처리기 (필수!)
└── sessions/           # 세션 데이터 (자동 생성)
```
## 필수패키지
```
# requirements.txt (필수 패키지만)

# 웹 프레임워크
fastapi==0.115.0
uvicorn==0.30.1
python-multipart==0.0.9

# HTTP 클라이언트
httpx==0.27.0

# 환경 변수
python-dotenv==1.2.1

# 오디오 처리
numpy==2.0.2
soundfile==0.13.1
```


## 🚀 실행 방법

### 1. 판단서버 실행
```bash
cd judgeTest
python judge_server.py
```

### 2. 통신서버 실행 (다른 터미널)
```bash
python app.py
```

### 3. 브라우저 접속
```
http://127.0.0.1:8000/
```

## 🔑 핵심 사항

### ⚠️ 절대 변경 금지
1. **processor.js 경로**: `/static/processor.js` (streaming.html 327번째 줄)
2. **FormData 필드명**: `sessionId`, `seq`, `chunk` (251-254번째 줄)
3. **AudioWorklet 이름**: `'audio-stream-processor'` (334번째 줄)
4. **파일 배치**: `processor.js`는 반드시 `static/` 폴더 안

### ✅ HTTPS 필수
- 배포 시 HTTPS 사용 (마이크 권한 필요)
- 개발: `localhost`만 HTTP 허용

## 📡 API 명세

### POST /start
```json
응답: {"sessionId": "uuid"}
```

### POST /upload-chunk
```json
요청: FormData {sessionId, seq, chunk}
응답: {"seq": int, "status": "Silent|Speech|Finished|Error", "text": string}
```

## 🎯 통신 흐름
```
프론트(HTML) → 통신서버(8000) → 판단서버(9000) → Whisper STT
```