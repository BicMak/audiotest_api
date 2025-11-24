import dataclasses

# ========== 설정 클래스 ==========
@dataclasses.dataclass
class AudioConfig:
    """오디오 설정"""
    SAMPLERATE: int = 16000
    SILENCE_THRESHOLD: int = 3
    EXIT_THRESHOLD: int = 10
    GAIN: float = 1.0
    VAD_THRESHOLD: float = 0.2
    WHISPER_MODEL: str = "whisper-1"
    WHISPER_LANGUAGE: str = "ko"
    NEG_THRESHOLD : float = 0.1
    MIN_SPEACH_DURATION_MS : int = 100

@dataclasses.dataclass
class ServerConfig:
    """서버 설정"""
    HOST: str = "127.0.0.1"
    PORT: int = 9000
    MIDDLE_SERVER_URL: str = None

@dataclasses.dataclass
class CORSConfig:
    """CORS 설정"""
    ALLOW_ORIGINS: list = dataclasses.field(default_factory=lambda: ["*"])
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list = dataclasses.field(default_factory=lambda: ["*"])
    ALLOW_HEADERS: list = dataclasses.field(default_factory=lambda: ["*"])

@dataclasses.dataclass
class PathConfig:
    """경로 설정"""
    SESSIONS_DIR: str = "sessions_b"
    INBOX_DIR: str = "inbox"
    TEMP_FILE_PREFIX: str = "temp_audio_"

@dataclasses.dataclass
class VADConfig:
    """VAD 모델 설정"""
    MONITORING: bool = True