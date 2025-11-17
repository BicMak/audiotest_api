```markdown
# 🎤 판단 서버 (Judge Server)

Raw PCM 청크를 받아서 VAD(음성 활동 감지) + Whisper STT를 수행하는 백엔드 API 서버

## 📋 주요 기능

1. **VAD (Voice Activity Detection)**: Silero VAD로 음성/무음 감지
2. **STT (Speech-to-Text)**: OpenAI Whisper API로 한국어 음성 인식
3. **상태 관리**: 세션별 음성 녹음 시작/종료 추적

## 🚀 실행 방법

### 환경 변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=sk-your-api-key-here
```

### 서버 실행
```bash
python judge_server.py
```

### CLI 테스트 모드
```bash
# 파일 경로 수정 후 (398번째 줄)
python judge_server.py
```

## 📡 API 명세

### 1. POST /start
세션 시작

**응답:**
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. POST /ingest-chunk
Raw PCM 청크 처리

**요청 (FormData):**
- `sessionId`: 세션 ID (string)
- `chunk`: Raw PCM 바이너리 (Int16, 16kHz, Mono)

**응답:**

**Case 1: 무음 또는 녹음 중 (204)**
```http
HTTP 204 No Content
```

**Case 2: 음성 종료 + STT 완료 (200)**
```json
{
  "status": "Finished",
  "text": "안녕하세요"
}
```

**Case 3: 에러 (500)**
```json
{
  "status": "Error",
  "text": null,
  "detail": "에러 상세"
}
```

## 🔧 핵심 설정

### AudioConfig (44-48번째 줄)
```python
SAMPLERATE = 16000          # 샘플레이트 (고정)
SILENCE_THRESHOLD = 3       # 무음 3회 → 녹음 종료
EXIT_THRESHOLD = 10         # 무음 10회 → 에러
```

### VAD 임계값 (73번째 줄)
```python
threshold=0.2  # 낮을수록 민감 (0.0~1.0)
```

## 📊 상태 다이어그램

```
[Silent] 무음 대기
    ↓ 음성 감지
[Speech] 녹음 중
    ↓ 무음 3회
[Finished] 녹음 완료 → Whisper STT
    ↓
프론트엔드로 텍스트 반환

특수 케이스:
[Silent] → 무음 10회 → [Error] 시스템 종료
```

## 🎯 처리 흐름

```
Raw PCM 수신 (Int16)
    ↓
Float32 변환 (-1.0 ~ 1.0)
    ↓
Silero VAD (음성 감지)
    ↓
음성 버퍼 누적
    ↓
무음 3회 감지 시
    ↓
WAV 저장 (temp_audio.wav)
    ↓
Whisper API 호출
    ↓
텍스트 반환
```

## 🔑 핵심 클래스

### VADModel (51-75번째 줄)
Silero VAD 모델 래퍼
- `get_speech_timestamps()`: 음성 구간 타임스탬프 반환

### _AudioActivityDetection (78-165번째 줄)
음성 활동 추적
- `is_recording`: 녹음 중 여부
- `speech_buffer`: 음성 데이터 버퍼
- `stop_count`: 연속 무음 카운트

## ⚙️ 의존성

```bash
pip install fastapi uvicorn python-multipart python-dotenv
pip install numpy soundfile silero-vad openai
pip install librosa  # CLI 테스트용 (선택)
```

## 🧪 CLI 테스트

오디오 파일로 VAD + STT 테스트 (398-470번째 줄)

**파일 경로 수정:**
```python
audio_file = "your-audio-file.mp3"  # 398번째 줄
```

**실행:**
```bash
python judge_server.py
```

**출력 예시:**
```
📂 파일 로딩 중: audio.mp3
📊 총 20개 청크 (각 0.5초)
[청크 #1/20] (0.0s ~ 0.5s)
  📊 Status: Silent
[청크 #5/20] (2.0s ~ 2.5s)
  📊 Status: Speech
🎤 음성 시작
[청크 #10/20] (4.5s ~ 5.0s)
✅ 음성 종료
  📝 Text: 안녕하세요
```

## 📝 주요 함수

### process_audio_chunk() (238-295번째 줄)
```python
def process_audio_chunk(session_id: str, audio_data, reset: bool = False) -> dict
```

**역할:**
1. VAD로 음성 감지
2. 음성 시작/종료 판단
3. 종료 시 Whisper STT 호출
4. 결과 반환

**반환값:**
```python
{
    "status": "Silent|Speech|Finished|Error|Reset",
    "text": "인식된 텍스트" or None
}
```

## ⚠️ 주의사항

1. **OpenAI API 키 필수**: `.env` 파일에 `OPENAI_API_KEY` 설정
2. **16kHz Int16 전용**: 다른 형식은 변환 필요
3. **단일 세션**: 현재 구현은 세션 격리 미지원 (168번째 줄)
4. **임시 파일**: `temp_audio.wav` 자동 생성/삭제

## 🔗 연동 서버

통신서버 (app.py)가 이 서버의 `/ingest-chunk`로 요청 전달
```
프론트엔드 → 통신서버(8000) → 판단서버(9000)
```
```