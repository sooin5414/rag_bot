"""
WhisperX 재실행 스크립트
- initial_prompt로 영어 문법 용어 힌트 추가
- 기존 JSON 삭제 후 재생성
"""
import whisperx
import gc
import torch
import json
import glob
import os

# =============================================================================
# 1. 설정
# =============================================================================

# 영어 문법 용어 힌트 (Whisper가 한글 음차로 잘못 인식하는 것 방지)
GRAMMAR_PROMPT = """
be동사, am, is, are, was, were, been, being,
have, has, had, having,
do, does, did,
will, would, shall, should, can, could, may, might, must,
현재완료, 과거완료, 미래완료, 진행형, 수동태, 능동태,
조동사, 전치사, 접속사, 관계대명사, 관계부사,
to부정사, 동명사, 분사, 현재분사, 과거분사,
주어, 목적어, 보어, 명사, 동사, 형용사, 부사
"""

AUDIO_FOLDER = "./youtube_playlist"
# cudnn 문제로 CPU 사용 (속도는 느리지만 안정적)
device = "cpu"
batch_size = 4
compute_type = "int8"

print(f"Device: {device}, Batch size: {batch_size}, Compute type: {compute_type}")

# =============================================================================
# 2. 기존 JSON 삭제 (스킵 - 모델 로드 후 삭제)
# =============================================================================

# JSON 삭제는 모델 로드 성공 후 진행
json_files = glob.glob(f"{AUDIO_FOLDER}/*.json")
print(f"\n기존 JSON 파일: {len(json_files)}개 (모델 로드 후 삭제 예정)")

# =============================================================================
# 3. WhisperX 모델 로드
# =============================================================================

print("WhisperX 모델 로딩...")
# asr_options로 initial_prompt 전달
asr_options = {
    "initial_prompt": GRAMMAR_PROMPT
}
# vad_method를 silero로 변경 (pyannote는 cudnn 필요)
model = whisperx.load_model(
    "large-v2",
    device,
    compute_type=compute_type,
    language="ko",
    asr_options=asr_options,
    vad_method="silero"  # pyannote 대신 silero 사용
)
print("모델 로드 완료\n")

# 모델 로드 성공 후 JSON 삭제
if json_files:
    print(f"기존 JSON 파일 {len(json_files)}개 삭제 중...")
    for f in json_files:
        os.remove(f)
        print(f"  삭제: {os.path.basename(f)}")
    print("삭제 완료\n")

# =============================================================================
# 4. MP3 파일 처리
# =============================================================================

mp3_files = sorted(glob.glob(f"{AUDIO_FOLDER}/*.mp3"))
print(f"처리할 파일: {len(mp3_files)}개\n")

for i, mp3_file in enumerate(mp3_files, 1):
    filename = os.path.basename(mp3_file)
    json_filename = mp3_file.replace('.mp3', '.json')

    print(f"[{i}/{len(mp3_files)}] {filename}")

    # Transcribe (initial_prompt는 load_model에서 설정됨)
    audio = whisperx.load_audio(mp3_file)
    result = model.transcribe(
        audio,
        batch_size=batch_size
    )

    # Align (타임스탬프 정확도 향상)
    print("  - Aligning...")
    model_a, metadata = whisperx.load_align_model(language_code="ko", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # JSON 저장
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  - 완료: {len(result['segments'])} segments")

    # Alignment 모델 메모리 해제
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

# =============================================================================
# 5. 정리
# =============================================================================

del model
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

print(f"\n전체 완료! {len(mp3_files)}개 파일 처리됨")
