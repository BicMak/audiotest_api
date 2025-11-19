# ============================================
# 통신서버 (app.py) - Raw PCM 패스스루 버전
# ============================================

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
from pathlib import Path
import json
import httpx

app = FastAPI()

BASE_DIR = Path(__file__).parent
SESS_BASE = BASE_DIR / "sessions"
SESS_BASE.mkdir(exist_ok=True)

# 정적 파일 제공
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "static" / "streaming.html")  # ← 🔥 수정!

@app.get("/monitoring.html")
def monitoring():
    return FileResponse(BASE_DIR / "monitoring.html")

# ---------- 판단 서버 설정 ----------
JUDGE_BASE_URL = "http://127.0.0.1:9000"
JUDGE_INGEST_CHUNK = f"{JUDGE_BASE_URL}/ingest-chunk"


# ---------- 헬퍼 ----------
def sess_dir(sid: str) -> Path:
    """세션별 디렉토리 생성 및 반환"""
    d = SESS_BASE / sid
    d.mkdir(parents=True, exist_ok=True)
    return d

def meta_path(sid: str) -> Path:
    """세션 메타데이터 파일 경로 반환"""
    return sess_dir(sid) / "meta.json"

def load_meta(sid: str) -> dict:
    """세션 메타데이터 로드"""
    p = meta_path(sid)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"state": "waiting", "had_speech": False}

def save_meta(sid: str, meta: dict):
    """세션 메타데이터 저장"""
    meta_path(sid).write_text(json.dumps(meta), encoding="utf-8")


async def send_to_judge(session_id: str, chunk_file: UploadFile) -> tuple[int, dict]:
    """
    판단 서버에 청크 그대로 전달 (패스스루)
    
    Args:
        session_id: 세션 ID
        chunk_file: 프론트엔드에서 받은 Raw PCM 청크
        
    Returns:
        (status_code, response_data)
    """
    try:
        # 🔥 핵심: 받은 파일을 그대로 판단서버로 전달
        chunk_data = await chunk_file.read()
        
        files = {
            "chunk": (chunk_file.filename, chunk_data, "application/octet-stream")
        }
        data = {
            "sessionId": session_id
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                JUDGE_INGEST_CHUNK,
                data=data,
                files=files
            )
        
        # 204: 계속 진행
        if resp.status_code == 204:
            return 204, {"status": "Silent", "text": None}
        
        # 200: Finished
        elif resp.status_code == 200:
            return 200, resp.json()
        
        # 500: Error
        elif resp.status_code == 500:
            return 500, resp.json()
        
        else:
            return 500, {"status": "Error", "text": None}
            
    except Exception as e:
        print(f"❌ 판단 서버 통신 에러: {e}")
        return 500, {"status": "Error", "text": None, "detail": str(e)}


def judge_status_to_frontend_status(judge_status: str, had_speech: bool) -> str:
    """
    판단서버 status를 프론트엔드 status로 변환
    
    Args:
        judge_status: "Silent" | "Finished" | "Error"
        had_speech: 이전에 음성이 감지된 적 있는지
        
    Returns:
        "Silent" | "Speech" | "Finished" | "Error"
    """
    if judge_status == "Finished":
        return "Finished"
    elif judge_status == "Error":
        return "Error"
    elif had_speech:
        return "Speech"  # 녹음 중
    else:
        return "Silent"  # 대기 중


# ---------- 라우트 ----------
@app.post("/start")
def start():
    """
    새 녹음 세션 시작
    
    Returns:
        {"sessionId": "uuid-string"}
    """
    sid = str(uuid4())
    save_meta(sid, {
        "state": "waiting",
        "had_speech": False
    })
    return {"sessionId": sid}


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
        
        async with httpx.AsyncClient(timeout=30.0) as client:  # 파일 모드는 시간 더 필요
            resp = await client.post(
                JUDGE_INGEST_CHUNK,
                data=data,
                files=files
            )
        
        # 204: 계속 진행 (스트리밍 모드)
        if resp.status_code == 204:
            return JSONResponse({
                "status": "Silent",
                "text": None
            }, status_code=200)
        
        # 200: Finished
        elif resp.status_code == 200:
            return JSONResponse(resp.json(), status_code=200)
        
        # 500: Error
        else:
            return JSONResponse(resp.json(), status_code=500)
            
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