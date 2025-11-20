# ============================================
# 통신서버 (app.py) - Raw PCM 패스스루 버전
# ============================================

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import httpx

app = FastAPI()

BASE_DIR = Path(__file__).parent

# 정적 파일 제공
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "static" / "streaming.html")

@app.get("/monitoring.html")
def monitoring():
    return FileResponse(BASE_DIR / "monitoring.html")

# ---------- 판단 서버 설정 ----------
JUDGE_BASE_URL = "http://127.0.0.1:9000"
JUDGE_START = f"{JUDGE_BASE_URL}/start"
JUDGE_INGEST_CHUNK = f"{JUDGE_BASE_URL}/ingest-chunk"


# ---------- 라우트 ----------
@app.post("/start")
async def start():
    """
    새 녹음 세션 시작 - 판단서버로 프록시
    
    Returns:
        {"sessionId": "uuid-string"}
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(JUDGE_START)
        
        if resp.status_code == 200:
            return JSONResponse(resp.json(), status_code=200)
        else:
            return JSONResponse({
                "error": "Failed to create session"
            }, status_code=500)
            
    except Exception as e:
        print(f"❌ 판단 서버 통신 에러: {e}")
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


@app.post("/ingest-chunk")
async def ingest_chunk(
    sessionId: str = Form(...),
    chunk: UploadFile = Form(...),
    mode: str = Form("chunk")
):
    """
    오디오 청크/파일 패스스루
    
    Args:
        sessionId: 세션 ID
        chunk: Raw PCM 청크 또는 WAV 파일
        mode: "chunk" (스트리밍) 또는 "file" (파일 전사)
        
    Returns:
        {
            "status": "Silent" | "Speech" | "Finished" | "Error",
            "text": str | null
        }
    """
    try:
        chunk_data = await chunk.read()
        
        files = {
            "chunk": (chunk.filename, chunk_data, "application/octet-stream")
        }
        data = {
            "sessionId": sessionId,
            "mode": mode
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                JUDGE_INGEST_CHUNK,
                data=data,
                files=files
            )
        
        # 판단서버 응답 그대로 전달
        return JSONResponse(resp.json(), status_code=resp.status_code)
            
    except Exception as e:
        print(f"❌ 판단 서버 통신 에러: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "Error",
            "text": None,
            "detail": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)