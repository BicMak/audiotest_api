#!/usr/bin/env python3
"""
로컬 VAD 테스트 스크립트
- 서버 없이 _AudioActivityDetection 클래스 직접 테스트
- 다중 유저 동시 테스트 (threading으로 시뮬레이션)
"""
import threading
import time
import random
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa
from silero_vad import load_silero_vad, get_speech_timestamps
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ========== Config ==========
@dataclass
class AudioConfig:
    SAMPLERATE: int = 16000
    SILENCE_THRESHOLD: int = 3      # 연속 무음 횟수 → 발화 종료
    EXIT_THRESHOLD: int = 10        # 연속 무음 횟수 → 에러
    GAIN: float = 3.0
    VAD_THRESHOLD: float = 0.2
    NEG_THRESHOLD: float = 0.1
    MIN_SPEECH_DURATION_MS: int = 250


@dataclass
class VADConfig:
    MONITORING: bool = False


# ========== 설정 ==========
AUDIO_CONFIG = AudioConfig()
VAD_CONFIG = VADConfig()
TEST_AUDIO_DIR = Path("test_audio/")
CHUNK_DURATION = 0.5  # 초
NUM_USERS = 3


# ========== VAD 모델 ==========
class VADModel:
    """VAD 모델 래퍼 클래스"""
    def __init__(self, audio_config: AudioConfig, vad_config: VADConfig) -> None:
        self.model = load_silero_vad()
        self.SAMPLERATE = audio_config.SAMPLERATE
        self.VAD_THRESHOLD = audio_config.VAD_THRESHOLD
        self.NEG_THRESHOLD = audio_config.NEG_THRESHOLD
        self.MIN_SPEECH_DURATION_MS = audio_config.MIN_SPEECH_DURATION_MS
        self.monitoring = vad_config.MONITORING

    def get_speech_timestamps(self, audio_data) -> list:
        """오디오 데이터에서 음성 구간의 타임스탬프를 반환"""
        if self.monitoring:
            print(f"[VAD] shape: {audio_data.shape}, range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        
        return get_speech_timestamps(
            audio_data,
            self.model,
            threshold=self.VAD_THRESHOLD,
            sampling_rate=self.SAMPLERATE,
            min_speech_duration_ms=self.MIN_SPEECH_DURATION_MS,
            neg_threshold=self.NEG_THRESHOLD,
        )


# ========== 음성 활동 감지 ==========
class _AudioActivityDetection:
    """음성 활동 감지 클래스"""
    def __init__(self, audio_config: AudioConfig):
        self.UserBuffer = dict()
        self.silence_threshold = audio_config.SILENCE_THRESHOLD
        self.exit_threshold = audio_config.EXIT_THRESHOLD
        self.lock = threading.Lock()

    def resetStream(self):
        """스트림 상태 초기화"""
        self.UserBuffer = dict()
        return {"audio": None, "status": "Reset"}

    def __call__(self, 
                 user_id: str,
                 speech_detected: list, 
                 audio_buffer: np.array) -> dict:
        """음성 데이터에서 화자 활동을 감지"""
        with self.lock:
            has_speech = len(speech_detected) > 0
            user_status = "Silent"
            user_audio = None

            # 버퍼 크기 체크
            if len(self.UserBuffer) > 100:
                raise Exception("버퍼가 가득참")
            
            if user_id not in self.UserBuffer:
                self.UserBuffer[user_id] = {
                    'is_recording': False,
                    'buffer': [],
                    'stop_count': 0
                }

            if has_speech:
                if not self.UserBuffer[user_id]['is_recording']:
                    self.UserBuffer[user_id]['is_recording'] = True
                    self.UserBuffer[user_id]['stop_count'] = 0
                    user_status = "Speech"
                    print(f"🎤 [{user_id}] 음성 시작")
                else:
                    user_status = "Speech"
                
                self.UserBuffer[user_id]['buffer'].append(audio_buffer)
                
                if self.UserBuffer[user_id]['stop_count'] > 0:
                    print(f"[{user_id}] 음성 재감지 → 무음 카운트 리셋")
                    self.UserBuffer[user_id]['stop_count'] = 0
                
            else:  # 무음
                if self.UserBuffer[user_id]['is_recording']:
                    zero_data = np.zeros_like(audio_buffer)
                    self.UserBuffer[user_id]['buffer'].append(zero_data)
                    self.UserBuffer[user_id]['stop_count'] += 1
                    user_status = "Speech"
                    
                    print(f"[{user_id}] 연속 무음: {self.UserBuffer[user_id]['stop_count']}/{self.silence_threshold}")
                    
                    if self.UserBuffer[user_id]['stop_count'] >= self.silence_threshold:
                        speech_data = np.concatenate(self.UserBuffer[user_id]["buffer"], axis=0)
                        self.UserBuffer[user_id]['is_recording'] = False
                        self.UserBuffer[user_id]['stop_count'] = 0
                        user_audio = speech_data
                        user_status = "Finished"
                        self.UserBuffer.pop(user_id)
                        print(f"✅ [{user_id}] 음성 종료 (길이: {len(speech_data)} samples)")
                        
                else:
                    self.UserBuffer[user_id]['stop_count'] += 1
                    if self.UserBuffer[user_id]['stop_count'] >= self.exit_threshold:
                        print(f"❌ [{user_id}] 연속 {self.exit_threshold}번 무음 → 에러")
                        user_status = "Error"
                    else:
                        user_status = "Silent"

            return {"audio": user_audio, "status": user_status}


# ========== 오디오 처리 함수 ==========
def process_audio_chunk(vad_model: VADModel,
                        detector: _AudioActivityDetection,
                        user_id: str,
                        audio_data: np.array) -> dict:
    """오디오 청크 처리 (서버 없이 직접 호출)"""
    # VAD 실행
    speech_timestamps = vad_model.get_speech_timestamps(audio_data)
    
    # 활동 감지
    result = detector(user_id, speech_timestamps, audio_data)
    
    # Whisper API로 음성 인식
    if result["audio"] is not None:
        result["text"] = whisper_transcribe(result["audio"], AUDIO_CONFIG.SAMPLERATE)
    else:
        result["text"] = None
    
    return result


def whisper_transcribe(audio_data: np.array, sample_rate: int) -> str:
    """Whisper API로 음성 인식"""
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio_data, sample_rate)
    
    try:
        with open(temp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
        return response.text
    finally:
        os.remove(temp_path)


# ========== 오디오 로드 유틸 ==========
def load_and_chunk_audio(wav_path: Path, sample_rate: int = 16000, chunk_duration: float = 0.5) -> list:
    """wav 파일을 청크로 쪼개기"""
    audio_data, sr = sf.read(wav_path, dtype='float32')
    
    # 스테레오 → 모노
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # 리샘플링
    if sr != sample_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
    
    # 청크로 쪼개기
    chunk_size = int(sample_rate * chunk_duration)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    # 무음 청크 추가 (발화 종료 감지용)
    silence = np.zeros(chunk_size, dtype=np.float32)
    for _ in range(5):
        chunks.append(silence)
    
    return chunks


def create_dummy_audio(sample_rate: int = 16000, chunk_duration: float = 0.5) -> list:
    """테스트용 더미 오디오 생성 (사인파)"""
    chunk_size = int(sample_rate * chunk_duration)
    
    # 2초짜리 440Hz 사인파
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    
    # 무음 청크 추가
    silence = np.zeros(chunk_size, dtype=np.float32)
    for _ in range(5):
        chunks.append(silence)
    
    return chunks


# ========== 테스트 함수 ==========
def test_single_user(vad_model: VADModel, 
                     detector: _AudioActivityDetection,
                     user_id: str, 
                     chunks: list) -> dict:
    """단일 유저 테스트"""
    result = None
    
    for i, chunk in enumerate(chunks):
        chunk_with_gain = chunk * AUDIO_CONFIG.GAIN
        result = process_audio_chunk(vad_model, detector, user_id, chunk_with_gain)
        
        print(f"📤 [{user_id}] chunk {i} → {result['status']}" + 
              (f" | 텍스트: {result['text']}" if result.get('text') else ""))
        
        if result["status"] in ["Finished", "Error"]:
            break
        
        time.sleep(0.05)
    
    return result


def test_multiple_users_interleaved(vad_model: VADModel,
                                    detector: _AudioActivityDetection,
                                    user_chunks: dict) -> dict:
    """여러 유저 청크를 섞어서 테스트"""
    chunk_indices = {user_id: 0 for user_id in user_chunks.keys()}
    results = {user_id: None for user_id in user_chunks.keys()}
    finished = {user_id: False for user_id in user_chunks.keys()}
    
    print("\n" + "=" * 50)
    print("🔀 청크 섞어서 전송 시작")
    print("=" * 50 + "\n")
    
    while not all(finished.values()):
        # 아직 안 끝난 유저 중 랜덤 선택
        active_users = [u for u, f in finished.items() if not f]
        if not active_users:
            break
        
        user_id = random.choice(active_users)
        idx = chunk_indices[user_id]
        
        if idx >= len(user_chunks[user_id]):
            finished[user_id] = True
            continue
        
        chunk = user_chunks[user_id][idx]
        chunk_with_gain = chunk * AUDIO_CONFIG.GAIN
        
        result = process_audio_chunk(vad_model, detector, user_id, chunk_with_gain)
        results[user_id] = result
        chunk_indices[user_id] += 1
        
        print(f"📤 [{user_id}] chunk {idx} → {result['status']}" + 
              (f" | 텍스트: {result['text']}" if result.get('text') else ""))
        
        if result["status"] in ["Finished", "Error"]:
            finished[user_id] = True
        
        time.sleep(0.02)
    
    return results


def test_concurrent_threads(vad_model: VADModel,
                            detector: _AudioActivityDetection,
                            user_chunks: dict) -> dict:
    """실제 멀티스레드로 동시 테스트"""
    results = {}
    threads = []
    
    def worker(user_id: str, chunks: list):
        result = test_single_user(vad_model, detector, user_id, chunks)
        results[user_id] = result
    
    print("\n" + "=" * 50)
    print("🧵 멀티스레드 동시 테스트 시작")
    print("=" * 50 + "\n")
    
    for user_id, chunks in user_chunks.items():
        t = threading.Thread(target=worker, args=(user_id, chunks))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    return results


# ========== 메인 ==========
def main():
    print("=" * 50)
    print("🧪 로컬 VAD 테스트 (서버 없이)")
    print("=" * 50)
    
    # 모델 초기화
    print("\n📦 VAD 모델 로딩...")
    vad_model = VADModel(AUDIO_CONFIG, VAD_CONFIG)
    detector = _AudioActivityDetection(AUDIO_CONFIG)
    print("✅ 모델 로딩 완료\n")
    
    # 오디오 파일 찾기
    TEST_AUDIO_DIR = Path(__file__).parent / "test_audio"        
    wav_files = list(TEST_AUDIO_DIR.glob("*.wav"))
    mp3_files = list(TEST_AUDIO_DIR.glob("*.mp3"))
    
    # mp3 → wav 변환
    if mp3_files:
        try:
            from pydub import AudioSegment
            for mp3_file in mp3_files:
                wav_path = mp3_file.with_suffix(".wav")
                if not wav_path.exists():
                    print(f"🔄 변환: {mp3_file.name} → {wav_path.name}")
                    audio = AudioSegment.from_mp3(mp3_file)
                    audio.export(wav_path, format="wav")
                wav_files.append(wav_path)
        except ImportError:
            print("⚠️  pydub 없음. mp3 변환 스킵")
    
    # 오디오 청크 준비
    user_chunks = {}
    
    if wav_files:
        print(f"📂 {len(wav_files)}개 wav 파일 발견\n")
        for i, wav_file in enumerate(wav_files[:NUM_USERS]):
            user_id = f"user_{i+1}"
            chunks = load_and_chunk_audio(wav_file, AUDIO_CONFIG.SAMPLERATE, CHUNK_DURATION)
            user_chunks[user_id] = chunks
            print(f"📁 [{user_id}] {wav_file.name}: {len(chunks)} 청크")
    else:
        print(f"⚠️  {TEST_AUDIO_DIR} 폴더에 오디오 파일 없음")
        print("→ 더미 오디오로 테스트\n")
        for i in range(NUM_USERS):
            user_id = f"user_{i+1}"
            user_chunks[user_id] = create_dummy_audio(AUDIO_CONFIG.SAMPLERATE, CHUNK_DURATION)
            print(f"📁 [{user_id}] 더미 오디오: {len(user_chunks[user_id])} 청크")
    
    # 테스트 방식 선택
    print("\n" + "-" * 50)
    print("테스트 방식 선택:")
    print("  1. 섞어서 순차 테스트 (interleaved)")
    print("  2. 멀티스레드 동시 테스트 (concurrent)")
    print("-" * 50)
    
    choice = input("선택 (1/2, 기본=1): ").strip() or "1"
    
    if choice == "2":
        results = test_concurrent_threads(vad_model, detector, user_chunks)
    else:
        results = test_multiple_users_interleaved(vad_model, detector, user_chunks)
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    for user_id, result in results.items():
        if result:
            status = result["status"]
            text = result.get("text", "없음")
            print(f"[{user_id}] 상태: {status} | 텍스트: {text}")
        else:
            print(f"[{user_id}] 결과 없음")
    
    # 버퍼 상태 확인
    print(f"\n🔍 남은 버퍼: {list(detector.UserBuffer.keys())}")


if __name__ == "__main__":
    main()