#!/usr/bin/env python3
"""
판단 서버 (Judge Server)
청크를 받아서 VAD 처리하고 status 반환
"""
from fastapi.responses import FileResponse  # ← 이거 추가
from uuid import uuid4

import dataclasses
from openai import OpenAI
import time
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import numpy as np
import librosa
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict
import shutil
import io
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

# FastAPI 앱
app = FastAPI()

# CORS 설정 (필수!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
SESS_BASE = BASE / "sessions_b"
SESS_BASE.mkdir(exist_ok=True)
INBOX = BASE / "inbox"
INBOX.mkdir(exist_ok=True)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
client = OpenAI(api_key=api_key)

# ========== 설정 ==========
@dataclasses.dataclass
class AudioConfig:
    """오디오 설정 상수"""
    SAMPLERATE = 16000
    SILENCE_THRESHOLD = 3
    EXIT_THRESHOLD = 10


# ========== VAD 모델 ==========
class VADModel:
    """
    음성을 감지하는 VAD 모델 래퍼 클래스 (private)
    생성하면 동시에 VAD 모델을 로드합니다.
    
    Attributes:
        model: 로드된 VAD 모델
    """
    def __init__(self, monitoring=False) -> None:
        self.model = load_silero_vad()
        self.SAMPLERATE = AudioConfig.SAMPLERATE
        self.monitoring = monitoring

    def get_speech_timestamps(self, audio_data) -> list:
        """
        오디오 데이터에서 음성 구간의 타임스탬프를 반환합니다.
        
        Args:
            audio_data (np.array): 오디오 신호 배열

        Returns:
            list: 감지된 음성 구간의 타임스탬프 리스트
        """
        if self.monitoring:
            print(f"[VAD] audio_data type: {type(audio_data)}")
            print(f"[VAD] audio_data dtype: {audio_data.dtype}")
            print(f"[VAD] audio_data shape: {audio_data.shape}")
            print(f"[VAD] audio_data range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
  
        return get_speech_timestamps(
            audio_data,
            self.model,
            threshold=0.2,
            sampling_rate=self.SAMPLERATE,
        )


# ========== 음성 활동 감지 ==========
class _AudioActivityDetection:
    """
    음성 데이터를 읽어 와서 화자가 대화를 하고 있는지 감시
    Status
    - 1. Silent: 무음 상태
    - 2. Speech: 음성 감지 상태
    - 3. Finished: 음성 녹음 종료 상태
    - 4. Error: 연속 무음으로 인한 시스템 종료 상태
    - 5. Reset: 스트림 상태 초기화 상태
    
    Attributes:
        is_recording:  현재 녹음중인 여부로 최초로 음성이 감지되면 True로 변경되고,
                       연속으로 무음이 silence_threshold번 감지되면 False로 변경됩니다.
        speech_buffer: 녹음된 음성 데이터를 저장하는 버퍼  
        stop_count: 연속 무음 카운트
        silence_threshold: 연속 무음으로 간주하는 임계값
        exit_threshold: 연속 무음으로 간주하여 시스템 종료하는 임계값
    
    Methods:
        resetStream(): 스트림 상태 초기화
        __call__(speech_detected, audio_buffer): 음성 데이터에서 화자 활동을 감지하고 녹음 시작/종료를 제어

    """
    def __init__(self, 
                 silence_threshold: int = AudioConfig.SILENCE_THRESHOLD,
                 exit_threshold: int = AudioConfig.EXIT_THRESHOLD):
        self.is_recording = False
        self.speech_buffer = []
        self.stop_count = 0
        self.silence_threshold = silence_threshold
        self.exit_threshold = exit_threshold

    def resetStream(self):
        """스트림 상태 초기화"""
        self.is_recording = False
        self.speech_buffer = []
        self.stop_count = 0
        return {"audio": None, "status": "Reset"}

    def __call__(self, 
                 speech_detected: list,
                 audio_buffer: np.array) -> dict:
        """
        음성 데이터에서 화자 활동을 감지하고 녹음 시작/종료를 제어합니다.

        Args:
            speech_detected (list): 감지된 음성 구간의 타임스탬프 리스트
            audio_buffer (np.array): 현재 오디오 버퍼 데이터
        Returns:
            dict: {
                "audio": 완성된 음성 데이터 배열 또는 None,
                "status": "Silent" | "Speech" | "Finished" | "Error",
                "decision": "start" | "end" | None
            }
        """
        has_speech = len(speech_detected) > 0
        user_status = "Silent"
        user_audio = None
        
        if has_speech:
            if not self.is_recording:
                self.is_recording = True
                self.stop_count = 0
                self.speech_buffer = []
                user_status = "Speech"
                decision = "start"  # 음성 시작!
                print("🎤 음성 시작 (decision: start)")
            else:
                user_status = "Speech"
            
            self.speech_buffer.append(audio_buffer)
            
            if self.stop_count > 0:
                print(f"음성 재감지 → 무음 카운트 리셋 ({self.stop_count} → 0)")
                self.stop_count = 0
            
        else:  # 무음
            if self.is_recording:
                zero_data = np.zeros_like(audio_buffer)
                self.speech_buffer.append(zero_data)
                self.stop_count += 1
                user_status = "Speech"
                
                print(f"연속 무음: {self.stop_count}/{self.silence_threshold}")
                
                if self.stop_count >= self.silence_threshold:
                    speech_data = np.concatenate(self.speech_buffer, axis=0)
                    self.is_recording = False
                    self.stop_count = 0
                    self.speech_buffer = []
                    user_audio = speech_data
                    user_status = "Finished"
                    decision = "end"  # 음성 종료!
                    print("✅ 음성 종료 (decision: end)")
                    
            else:
                self.stop_count += 1
                if self.stop_count >= self.exit_threshold:
                    print(f"❌ 연속 {self.exit_threshold}번 무음으로 시스템 종료")
                    user_audio = None
                    user_status = "Error"
                else:
                    user_status = "Silent"

        return {"audio": user_audio, "status": user_status}


# 세션별 이벤트 체커 (각 세션마다 독립적인 상태 유지)
session_event_checkers = _AudioActivityDetection()
_vad_model = VADModel(monitoring=False)


# ========== 핵심 함수: 오디오 청크 처리 ==========
def process_audio_chunk(session_id: str, audio_data, reset: bool = False) -> dict:
    """
    실시간 오디오 청취 및 텍스트 변환 내부 함수
    
    Status
    - 1. Silent: 무음 상태
    - 2. Speech: 음성 감지 상태
    - 3. Finished: 음성 녹음 종료 상태
    - 4. Error: 연속 무음으로 인한 시스템 종료 상태
    - 5. Reset: 스트림 상태 초기화 상태
    
    Args:
        session_id (str): 세션 ID
        audio_data (np.array): 오디오 신호 배열
        reset (bool): 스트림 상태 초기화 여부
    Returns:
        dict: {
            "status": "Silent" | "Speech" | "Finished" | "Error" | "Reset",
            "text": 변환된 텍스트 또는 None
        }
    """
    
    event_checker = session_event_checkers
    vad_model = _vad_model
    
    result_status = None
    transcript_text = None

    if reset:
        result = event_checker.resetStream()
        return {"status": result["status"], "text": None}

    if audio_data is not None:
        speech_timestamps = vad_model.get_speech_timestamps(audio_data)
        result = event_checker(speech_timestamps, audio_data)
        
        result_status = result["status"]
                
        if result["audio"] is not None:
            # 임시 파일로 저장
            sf.write("temp_audio.wav", result["audio"], samplerate=AudioConfig.SAMPLERATE)

            # OpenAI Whisper API로 전사
            with open("temp_audio.wav", "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )

            transcript_text = response.text
            print(f"📝 인식된 텍스트: {transcript_text}")

        elif result["status"] == "Error":
            transcript_text = None

        elif result["status"] in ["Speech", "Silent"]:
            transcript_text = None

        elif result["status"] == "Reset":
            transcript_text = None
    else:
        result_status = "silent"
        transcript_text = None
                    
    return {"status": result_status, "text": transcript_text}   
                
# ========== CLI 테스트 모드 ==========
if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("🎤 오디오 청크 테스트 (0.5초 단위)")
    print("=" * 60)
    
    # 오디오 파일 경로 입력
    audio_file = "D:/007.Portfolio/audioTest/judgeTest/GIK_Listening_1_Track 085.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
        sys.exit(1)
    
    # 오디오 파일 로드
    print(f"\n📂 파일 로딩 중: {audio_file}")
    audio_data, sr = librosa.load(
        audio_file, 
        sr=16000  # 샘플레이트 지정 (리샘플링 포함)
        # mono=True 옵션은 기본값이라 생략해도 됨
    )
    # 0.5초 청크 크기 계산
    chunk_size = int(AudioConfig.SAMPLERATE * 0.5)  # 8000 샘플
    total_chunks = int(np.ceil(len(audio_data) / chunk_size))
    
    print(f"\n📊 총 {total_chunks}개 청크 (각 0.5초)")
    print("=" * 60)
    
    # 세션 ID
    session_id = "test-session-" + str(uuid4())[:8]
    
    # 청크별 처리
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(audio_data))
        
        # 청크 추출
        chunk = audio_data[start_idx:end_idx]
        
        # 마지막 청크 패딩 (8000 샘플 맞추기)
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        # VAD + STT 처리
        result = process_audio_chunk(session_id, chunk)
        
        # 결과 출력
        print(f"\n[청크 #{i+1}/{total_chunks}] ({start_idx/AudioConfig.SAMPLERATE:.1f}s ~ {end_idx/AudioConfig.SAMPLERATE:.1f}s)")
        print(f"  📊 Status: {result['status']}")
        
        if result['text']:
            print(f"  📝 Text: {result['text']}")
            print("\n" + "=" * 60)
            print("✅ 음성 인식 완료!")
            break
        
        # 0.5초 대기 (실제 스트리밍 시뮬레이션)
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("🏁 테스트 완료!")