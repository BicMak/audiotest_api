# tts_server.py
from openai import OpenAI
from fastapi import FastAPI, Body
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import time
import os

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 클라이언트
client = OpenAI()

@app.post("/generate-speech/")
async def generate_speech(request_text: str = Body(..., embed=True)):
    """텍스트를 음성으로 변환"""
    try:
        temp_path = f"tts_{time.time()}.wav"
        
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice= "coral",
            input=request_text,
            instructions="Speak in a natural tone.",
            response_format="wav"
        ) as response:
            response.stream_to_file(temp_path)
        
        # 파일 읽어서 반환
        with open(temp_path, "rb") as f:
            audio_data = f.read()
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return Response(
            content=audio_data,
            media_type="audio/wav"
        )
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.get("/")
def root():
    return {"message": "TTS API - POST /generate-speech/"}

if __name__ == "__main__":
    import uvicorn
    print("🔊 TTS 서버 시작: http://127.0.0.1:9002")
    uvicorn.run(app, host="127.0.0.1", port=9002)