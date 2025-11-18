// ============================================
// processor.js - AudioWorklet 오디오 처리기
// ============================================
// 
// 역할: 마이크 입력을 받아서 0.5초 단위 청크로 만들어 전송
// 
// 데이터 흐름:
// [마이크] → [AudioWorklet] → [청크 버퍼링] → [postMessage] → [HTML]
//
// 주의사항:
// - AudioWorklet은 별도 스레드에서 실행됨 (DOM 접근 불가)
// - process()는 약 128 샘플 단위로 호출됨
// - sampleRate는 AudioWorklet 전역 변수로 자동 제공됨
// ============================================

class AudioStreamProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // ========================================
        // 설정값
        // ========================================
        // sampleRate: AudioWorklet 전역 변수 (브라우저 기본값, 보통 48000Hz)
        // 서버에서 16kHz로 리샘플링하므로 여기선 원본 그대로 전송
        
        this.targetLength = Math.floor(sampleRate * 0.5);  // 0.5초 분량 샘플 수
        // 예: 48000Hz × 0.5초 = 24000 샘플
        // 예: 16000Hz × 0.5초 = 8000 샘플
        
        // ========================================
        // 버퍼 초기화
        // ========================================
        this.sampleBuffer = new Int16Array(this.targetLength);  // 청크 버퍼
        this.sampleCursor = 0;  // 현재 버퍼 위치
        this.isActive = true;   // 활성 상태 플래그

        // ========================================
        // 메시지 리스너 (HTML과 통신)
        // ========================================
        this.port.onmessage = (e) => {
            if (e.data === 'stop') {
                this.cleanup();
            }
        };
    }
    
    // ========================================
    // 리소스 정리
    // ========================================
    cleanup() {
        this.isActive = false;
        this.sampleBuffer = null;
        this.sampleCursor = 0;
        console.log('AudioStreamProcessor 정리 완료');
    }

    // ========================================
    // 오디오 처리 (메인 루프)
    // ========================================
    // 브라우저가 약 128 샘플 단위로 호출
    // inputs[0][0]: Float32Array (모노 채널, -1.0 ~ 1.0)
    process(inputs, outputs, parameters) {
        // 비활성 상태면 종료
        if (!this.isActive) return false;
        
        // 입력 검증
        const input = inputs[0];
        if (!input || input.length === 0) return true;
        
        const channel = input[0];  // 모노 채널
        if (!channel) return true;

        // ========================================
        // 샘플 변환 및 버퍼링
        // ========================================
        for (let i = 0; i < channel.length; i++) {
            // Float32 (-1.0 ~ 1.0) → Int16 (-32768 ~ 32767) 변환
            let s = Math.max(-1, Math.min(1, channel[i]));  // 클리핑
            let int16Sample = Math.round(s * 32767);
            this.sampleBuffer[this.sampleCursor] = Math.max(-32768, Math.min(32767, int16Sample));
            this.sampleCursor++;

            // ========================================
            // 버퍼가 가득 차면 전송
            // ========================================
            if (this.sampleCursor >= this.targetLength) {
                // 버퍼 복사 (원본 보존)
                const bufferToSend = new Int16Array(this.sampleBuffer);
                
                // HTML로 전송 (ArrayBuffer 형태)
                // Transferable로 전송해서 복사 오버헤드 제거
                this.port.postMessage(bufferToSend.buffer, [bufferToSend.buffer]);
                
                // 버퍼 리셋
                this.sampleBuffer = new Int16Array(this.targetLength);
                this.sampleCursor = 0;
            }
        }

        return this.isActive;  // true: 계속, false: 종료
    }
}

// ========================================
// AudioWorklet 등록
// ========================================
// HTML에서 new AudioWorkletNode(ctx, 'audio-stream-processor')로 사용
registerProcessor('audio-stream-processor', AudioStreamProcessor);