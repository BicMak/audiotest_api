#!/usr/bin/env python3
"""
판단 서버 (Judge Server)
청크를 받아서 VAD 처리하고 status 반환
"""
import librosa
import requests

from uuid import uuid4
import asyncio
import dataclasses
import shutil
from openai import OpenAI
import time
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import numpy as np
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import json
from collections import deque

from judge_config import AudioConfig,ServerConfig,CORSConfig,PathConfig,VADConfig

active_sessions = dict()

load_dotenv()

# ========== Config 로더 ==========
def load_config(config_path: str = "config.json"):
    """JSON 설정 파일 로드"""
    config_file = Path(config_path)
    
    if config_file.exists():
        print(f"✅ 설정 파일 로드: {config_path}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        print(f"⚠️  설정 파일 없음. 기본값 사용: {config_path}")
        config_data = {}
    
    # AudioConfig
    audio_conf = config_data.get("audio", {})
    audio_config = AudioConfig(
        SAMPLERATE=audio_conf.get("samplerate", 16000),
        SILENCE_THRESHOLD=audio_conf.get("silence_threshold", 3),
        EXIT_THRESHOLD=audio_conf.get("exit_threshold", 10),
        GAIN=audio_conf.get("gain", 3.0),
        VAD_THRESHOLD=audio_conf.get("vad_threshold", 0.2),
        WHISPER_MODEL=audio_conf.get("whisper_model", "whisper-1"),
        WHISPER_LANGUAGE=audio_conf.get("whisper_language", "ko"),
        NEG_THRESHOLD=audio_conf.get("neg_threshold", 0.1),
        MIN_SPEACH_DURATION_MS=audio_conf.get("min_speech_duration_ms", 250)
    )
    
    # ServerConfig
    server_conf = config_data.get("server", {})
    server_config = ServerConfig(
        HOST=server_conf.get("host", "127.0.0.1"),
        PORT=server_conf.get("port", 9000),
        MIDDLE_SERVER_URL = server_conf.get("middle_server_url",None)
    )
    
    # CORSConfig
    cors_conf = config_data.get("cors", {})
    cors_config = CORSConfig(
        ALLOW_ORIGINS=cors_conf.get("allow_origins", ["*"]),
        ALLOW_CREDENTIALS=cors_conf.get("allow_credentials", True),
        ALLOW_METHODS=cors_conf.get("allow_methods", ["*"]),
        ALLOW_HEADERS=cors_conf.get("allow_headers", ["*"])
    )
    
    # PathConfig
    path_conf = config_data.get("paths", {})
    path_config = PathConfig(
        SESSIONS_DIR=path_conf.get("sessions_dir", "sessions_b"),
        INBOX_DIR=path_conf.get("inbox_dir", "inbox"),
        TEMP_FILE_PREFIX=path_conf.get("temp_file_prefix", "temp_audio_")
    )
    
    # VADConfig
    vad_conf = config_data.get("vad", {})
    vad_config = VADConfig(
        MONITORING=vad_conf.get("monitoring", False)
    )
    
    return audio_config, server_config, cors_config, path_config, vad_config


# ========== 설정 로드 ==========
AUDIO_CONFIG, SERVER_CONFIG, CORS_CONFIG, PATH_CONFIG, VAD_CONFIG = load_config()

# FastAPI 앱
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG.ALLOW_ORIGINS,
    allow_credentials=CORS_CONFIG.ALLOW_CREDENTIALS,
    allow_methods=CORS_CONFIG.ALLOW_METHODS,
    allow_headers=CORS_CONFIG.ALLOW_HEADERS,
)

# 경로 설정
BASE = Path(__file__).parent
SESS_BASE = BASE / PATH_CONFIG.SESSIONS_DIR
SESS_BASE.mkdir(exist_ok=True)
INBOX = BASE / PATH_CONFIG.INBOX_DIR
INBOX.mkdir(exist_ok=True)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
client = OpenAI(api_key=api_key)


# ========== VAD 모델 ==========
class VADModel:
    """VAD 모델 래퍼 클래스"""
    def __init__(self, audio_config: AudioConfig, vad_config: VADConfig) -> None:
        self.model = load_silero_vad()
        self.SAMPLERATE = audio_config.SAMPLERATE
        self.VAD_THRESHOLD = audio_config.VAD_THRESHOLD
        self.monitoring = vad_config.MONITORING
        self.NEG_THRESHOLD = audio_config.NEG_THRESHOLD
        self.MIN_SPEECH_DURATION_MS = audio_config.MIN_SPEACH_DURATION_MS

    def get_speech_timestamps(self, audio_data) -> list:
        """오디오 데이터에서 음성 구간의 타임스탬프를 반환"""
        if self.monitoring:
            print(f"[VAD] audio_data type: {type(audio_data)}")
            print(f"[VAD] audio_data dtype: {audio_data.dtype}")
            print(f"[VAD] audio_data shape: {audio_data.shape}")
            print(f"[VAD] audio_data range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
  
        return get_speech_timestamps(
            audio_data,
            self.model,
            threshold=self.VAD_THRESHOLD,
            sampling_rate=self.SAMPLERATE,
            min_speech_duration_ms = self.MIN_SPEECH_DURATION_MS,
            neg_threshold = self.NEG_THRESHOLD,
        )


# ========== 음성 활동 감지 ==========
class _AudioActivityDetection:
    """음성 활동 감지 클래스"""
    def __init__(self, audio_config: AudioConfig):
        self.UserBuffer = dict()
        self.silence_threshold = audio_config.SILENCE_THRESHOLD
        self.exit_threshold = audio_config.EXIT_THRESHOLD

    def resetStream(self):
        self.UserBuffer = dict()
        """스트림 상태 초기화"""

        return {"audio": None, "status": "Reset"}

    def __call__(self, 
                 user_id:str ,
                 speech_detected: list, 
                 audio_buffer: np.array,) -> dict:
        """음성 데이터에서 화자 활동을 감지"""
        has_speech = len(speech_detected) > 0
        user_status = "Silent"
        user_audio = None

        # 1. 우선 dict 버퍼가 크기를 초과 하지 않는지 테스트
        if len(self.UserBuffer) > 100:
            raise Exception("버퍼가 가득차서...") 
        else:
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
                print("🎤 음성 시작")
            else:
                user_status = "Speech"
            
            self.UserBuffer[user_id]['buffer'].append(audio_buffer)
            
            if self.UserBuffer[user_id]['stop_count']> 0:
                print(f"음성 재감지 → 무음 카운트 리셋 ({self.UserBuffer[user_id]['stop_count']} → 0)")
                self.UserBuffer[user_id]['stop_count'] = 0
            
        else:  # 무음
            if self.UserBuffer[user_id]['is_recording']:
                zero_data = np.zeros_like(audio_buffer)
                self.UserBuffer[user_id]['buffer'].append(zero_data)
                self.UserBuffer[user_id]['stop_count'] += 1
                user_status = "Speech"
                
                print(f"연속 무음: {self.UserBuffer[user_id]['stop_count']}/{self.silence_threshold}")
                
                if self.UserBuffer[user_id]['stop_count'] >= self.silence_threshold:
                    speech_data = np.concatenate(self.UserBuffer[user_id]["buffer"], axis=0)
                    self.UserBuffer[user_id]['is_recording'] = False
                    self.UserBuffer[user_id]['stop_count']= 0
                    user_audio = speech_data
                    user_status = "Finished"
                    self.UserBuffer.pop(user_id)
                    print("✅ 음성 종료")
                    
            else:
                self.UserBuffer[user_id]['stop_count'] += 1
                if self.UserBuffer[user_id]['stop_count'] >= self.exit_threshold:
                    print(f"❌ 연속 {self.exit_threshold}번 무음으로 시스템 종료")
                    user_audio = None
                    user_status = "Error"
                else:
                    user_status = "Silent"

        return {"audio": user_audio, "status": user_status}


# 세션 상태 및 VAD 모델 초기화
session_states = deque(maxlen=100) 
_vad_model = VADModel(AUDIO_CONFIG, VAD_CONFIG)
_audio_activate_model = _AudioActivityDetection(AUDIO_CONFIG)
# =============================================

def send_to_middle_server(status: str, text: str):
    """중간 서버로 텍스트 전송 (동기 방식)"""
    # check middle server url
    if(SERVER_CONFIG.MIDDLE_SERVER_URL is None):
        raise ValueError("MIDDLE_SERVER_URL이 설정되지 않았습니다.")
    else:
        middle_server_url = SERVER_CONFIG.MIDDLE_SERVER_URL

    payload = {
        "Status": status,
        "text": text,
    }
    
    try:
        response = requests.post(middle_server_url, json=payload, timeout=5)
        print(f"✅ 중간서버 전송 완료: {response.status_code}")
        return
    except Exception as e:
        print(f"❌ 중간서버 전송 실패: {e}")
        return


# ========== 핵심 함수: 오디오 청크 처리 ==========
async def process_audio_chunk(session_id: str, 
                              user_id: str, 
                              audio_data, 
                              reset: bool = False) -> dict:
    """실시간 오디오 청취 및 텍스트 변환"""
    
    # 중복 체크
    # 첫 청크면 등록, 이미 있으면 그냥 진행
    if session_id not in session_states:
        session_states.append(session_id)

    vad_model = _vad_model
    audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

    
    # 새 세션 등록 (maxlen=100 넘으면 자동으로 가장 오래된 것 제거)
    session_states.append(session_id)

    vad_model = _vad_model
    audio_data = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

    event_checker = _audio_activate_model  
    
    result_status = None
    transcript_text = None

    if reset:
        result = event_checker.resetStream()
        return {"status": result["status"], "text": None}

    if audio_data is not None:
        speech_timestamps = vad_model.get_speech_timestamps(audio_data)
        result = event_checker(user_id, speech_timestamps, audio_data)  # await 제거
        
        result_status = result["status"]
                
        if result["audio"] is not None:
            temp_file_name = f"{PATH_CONFIG.TEMP_FILE_PREFIX}{session_id}_{time.time()}.wav"
            
            await asyncio.to_thread(
                sf.write, 
                temp_file_name, 
                result["audio"], 
                AUDIO_CONFIG.SAMPLERATE
            )
            
            def transcribe_sync():
                with open(temp_file_name, "rb") as audio_file:
                    return client.audio.transcriptions.create(
                        model=AUDIO_CONFIG.WHISPER_MODEL,
                        file=audio_file,
                        language=AUDIO_CONFIG.WHISPER_LANGUAGE
                    )

            response = await asyncio.to_thread(transcribe_sync)
            transcript_text = response.text
            
            os.makedirs("audio_data", exist_ok=True)
            save_path = f"audio_data/{session_id}_{time.time()}.wav"
            await asyncio.to_thread(shutil.copy, temp_file_name, save_path)            
            await asyncio.to_thread(os.remove, temp_file_name)
                        
            print(f"📝 인식된 텍스트: {transcript_text}")

        elif result["status"] in ["Error", "Speech", "Silent", "Reset"]:
            transcript_text = None

    else:
        result_status = "silent"
        transcript_text = None
        
    if result_status in ["Finished", "Error"]:
        print(f"세션 {session_id} 상태 정리.")
        if session_id in session_states:
            session_states.remove(session_id)

    return {"status": result_status, "text": transcript_text}

# ========== FastAPI 라우트 ==========
@app.post("/start")
def start(userId: str = Form(...)):  # 유저ID 추가
    """새 세션 시작 (API 호환성 유지)"""
    sid = str(uuid4())
    active_sessions[sid] = {
        "userId": userId,
        "createdAt": time.time()
    }
    return {"sessionId": sid}

@app.post("/ingest-chunk")
async def ingest_chunk(
    userId: str = Form(...),
    sessionId: str = Form(...),
    chunk: UploadFile = Form(...),
    mode: str = Form("chunk")  # "chunk" 또는 "file"
):
    """청크/파일 수신 → VAD 처리 또는 직접 전사 → 응답 반환"""
    #함수 시작전에 무조껀 session ID 중복검사를 중복이면 에러로 반환함
    if sessionId not in active_sessions:
        return JSONResponse({
            "status": "Error",
            "text": None,
        }, status_code=400)    
    
    try:
        chunk_data = await chunk.read()
        print(f"📥 [판단] 세션: {sessionId[:8]}... | 모드: {mode} | 크기: {len(chunk_data)} bytes")
        
        # ========== 파일 모드: 바로 Whisper 전사 ==========
        if mode == "file":
            temp_path = f"{PATH_CONFIG.TEMP_FILE_PREFIX}{sessionId}_{time.time()}.wav"
            
            with open(temp_path, "wb") as f:
                f.write(chunk_data)
            
            def transcribe_sync():
                with open(temp_path, "rb") as audio_file:
                    return client.audio.transcriptions.create(
                        model=AUDIO_CONFIG.WHISPER_MODEL,
                        file=audio_file,
                        language=AUDIO_CONFIG.WHISPER_LANGUAGE
                    )
            
            response = await asyncio.to_thread(transcribe_sync)
            await asyncio.to_thread(os.remove, temp_path)
            
            print(f"📝 [파일모드] 인식된 텍스트: {response.text}")
            
            return JSONResponse({
                "status": "Finished",
                "text": response.text
            }, status_code=200)
        
        # ========== 청크 모드: VAD 처리 ==========
        else:
            audio_array = np.frombuffer(chunk_data, dtype=np.int16)
            audio_data = audio_array.astype(np.float32) / 32768.0
            
            print(f"🔄 [판단] 샘플 수: {len(audio_data)} | 범위: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            audio_data = audio_data * AUDIO_CONFIG.GAIN
            result = await process_audio_chunk(sessionId,userId, audio_data)
            
            print(f"🎯 [판단] VAD 결과: {result['status']}")
            
            if result["status"] == "Error":
                await asyncio.to_thread(
                    send_to_middle_server, 
                    result['status'], 
                    None
                )
                return JSONResponse({
                    "status": "Error",
                    "text": None
                }, status_code=500)

            elif result["status"] == "Finished":
                await asyncio.to_thread(
                    send_to_middle_server, 
                    result['status'], 
                    result['text']
                )
                return JSONResponse({
                    "status": "Finished",
                    "text": result["text"]
                }, status_code=200)

            elif result["status"] == "Speech":
                return JSONResponse({
                    "status": "Speech",
                    "text": None
                }, status_code=200)
            
            else: #Silent
                return JSONResponse({
                    "status": "Silent",
                    "text": None
                }, status_code=200)

    except Exception as e:
        print(f"❌ 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "Error",
            "text": None,
        }, status_code=500)

# ========== CLI 모드 ==========
if __name__ == '__main__':
    import uvicorn
    print(f"🚀 판단 서버 시작: {SERVER_CONFIG.HOST}:{SERVER_CONFIG.PORT}")
    uvicorn.run(app, host=SERVER_CONFIG.HOST, port=SERVER_CONFIG.PORT)