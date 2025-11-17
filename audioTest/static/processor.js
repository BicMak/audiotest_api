// processor.js
// AudioWorkletProcessor를 상속받아 오디오 처리 로직 구현
class AudioStreamProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 서버에서 16kHz를 기대하므로 리샘플링을 가정하고 16000으로 설정
        this.sampleRate = 16000; 
        // 0.5초(500ms)당 16000 * 0.5 = 8000 샘플
        this.targetLength = 8000; 
        // Int16 (2바이트) 배열로 데이터를 저장
        this.sampleBuffer = new Int16Array(this.targetLength);
        this.sampleCursor = 0;
        this.isActive = true;  // 🔥 활성 상태 플래그 추가
        
        // 🔥 메시지 리스너 추가 (정리 명령 수신용)
        this.port.onmessage = (e) => {
            if (e.data === 'stop') {
                this.cleanup();
            }
        };
    }
    
    // 🔥 정리 메서드 추가
    cleanup() {
        this.isActive = false;
        this.sampleBuffer = null;
        this.sampleCursor = 0;
        console.log('AudioStreamProcessor 정리 완료');
    }

    // 오디오 처리 메서드 (약 128개의 샘플 단위로 호출됨)
    process(inputs, outputs, parameters) {
        // 🔥 비활성 상태면 처리 중단
        if (!this.isActive) {
            return false;  // false 반환하면 processor 종료
        }
        
        // 첫 번째 입력 채널 (마이크)만 사용
        const input = inputs[0];
        if (input.length === 0) return true;
        
        const channel = input[0]; // 모노 가정 (Float32Array)

        for (let i = 0; i < channel.length; i++) {
            // Float32 (브라우저 기본, -1.0 ~ 1.0)를 Int16 (서버 VAD 선호, -32768 ~ 32767)으로 변환
            let s = Math.max(-1, Math.min(1, channel[i]));
            // Int16 범위로 변환: s * 32768
            let int16Sample = s < 0 ? s * 0x8000 : s * 0x7FFF;

            this.sampleBuffer[this.sampleCursor] = int16Sample;
            this.sampleCursor++;

            // 버퍼가 0.5초 분량(8000 샘플)에 도달하면 전송 준비
            if (this.sampleCursor >= this.targetLength) {
                // 🔥 활성 상태 체크
                if (!this.isActive) break;
                
                try {
                    // ArrayBuffer 복사 후 메인 스레드로 전송 (Transferable ArrayBuffer 사용)
                    // 이 동작은 데이터를 복사하는 대신 소유권을 넘겨주므로 효율적입니다.
                    this.port.postMessage(this.sampleBuffer.buffer, [this.sampleBuffer.buffer]);
                } catch (error) {
                    console.error('버퍼 전송 실패:', error);
                }
                
                // 버퍼 초기화 및 새 버퍼 할당
                this.sampleBuffer = new Int16Array(this.targetLength);
                this.sampleCursor = 0;
            }
        }

        // 🔥 활성 상태에 따라 반환값 결정
        return this.isActive;  // false면 processor 종료
    }
}

// AudioWorkletNode에서 사용할 이름으로 등록
registerProcessor('audio-stream-processor', AudioStreamProcessor);