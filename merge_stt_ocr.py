"""
STT JSON과 OCR JSON을 타임스탬프 기반으로 합치는 스크립트

STT: youtube_playlist/*.json (segments: [{start, end, text, words}])
OCR: ocr_results/*_ocr.json (ocr_data: [{timestamp, texts}])

결과: merged_data/*.json (segments에 screen_text 필드 추가)
"""

import json
import unicodedata
from pathlib import Path

# 경로 설정
STT_DIR = Path("youtube_playlist")
OCR_DIR = Path("ocr_results")
OUTPUT_DIR = Path("merged_data")


def normalize_filename(filename: str) -> str:
    """파일명 정규화 (유니코드 NFC + 파이프 문자 통일)"""
    # NFC 정규화
    normalized = unicodedata.normalize('NFC', filename)
    # NFD도 시도
    normalized_nfd = unicodedata.normalize('NFD', filename)
    # 전각 파이프(｜)를 반각(|)으로 통일
    normalized = normalized.replace('｜', '|')
    # 전각 슬래시(⧸)를 반각(/)으로 통일
    normalized = normalized.replace('⧸', '/')
    return normalized


def fuzzy_match_filename(stt_name: str, ocr_name: str) -> bool:
    """파일명 퍼지 매칭 - 유니코드 정규화 차이 무시"""
    # 기본 정규화
    stt_norm = normalize_filename(stt_name)
    ocr_norm = normalize_filename(ocr_name)

    if stt_norm == ocr_norm:
        return True

    # NFD로도 비교
    stt_nfd = unicodedata.normalize('NFD', stt_name).replace('｜', '|').replace('⧸', '/')
    ocr_nfd = unicodedata.normalize('NFD', ocr_name).replace('｜', '|').replace('⧸', '/')

    if stt_nfd == ocr_nfd:
        return True

    # 숫자 접두사로 매칭 (01_, 02_ 등)
    import re
    stt_prefix = re.match(r'^(\d+)_', stt_name)
    ocr_prefix = re.match(r'^(\d+)_', ocr_name)

    if stt_prefix and ocr_prefix and stt_prefix.group(1) == ocr_prefix.group(1):
        # 같은 번호면 매칭으로 간주
        return True

    return False


def find_matching_ocr_file(stt_filename: str, ocr_files: list) -> Path | None:
    """STT 파일에 매칭되는 OCR 파일 찾기"""
    stt_name = stt_filename.replace('.json', '')

    for ocr_file in ocr_files:
        ocr_name = ocr_file.name.replace('_ocr.json', '')
        if fuzzy_match_filename(stt_name, ocr_name):
            return ocr_file
    return None


def merge_stt_ocr(stt_data: dict, ocr_data: dict) -> dict:
    """STT와 OCR 데이터를 타임스탬프 기반으로 합치기"""
    segments = stt_data.get('segments', [])
    ocr_items = ocr_data.get('ocr_data', [])

    # 각 segment에 screen_text 필드 추가
    for segment in segments:
        start = segment['start']
        end = segment['end']

        # 해당 시간 범위에 속하는 OCR 텍스트 찾기
        matching_texts = []
        for ocr_item in ocr_items:
            ocr_time = ocr_item['timestamp']
            # OCR 타임스탬프가 segment 범위 내에 있으면 매칭
            if start <= ocr_time <= end:
                matching_texts.extend(ocr_item['texts'])

        # screen_text 필드 추가 (매칭된 텍스트가 있으면)
        if matching_texts:
            segment['screen_text'] = matching_texts

    # 메타데이터 추가
    result = {
        'video_url': ocr_data.get('video_url', ''),
        'segments': segments,
        'word_segments': stt_data.get('word_segments', [])
    }

    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # OCR 파일 목록
    ocr_files = list(OCR_DIR.glob('*_ocr.json'))
    print(f"OCR 파일 수: {len(ocr_files)}")

    # STT 파일 처리
    stt_files = sorted(STT_DIR.glob('*.json'))
    print(f"STT 파일 수: {len(stt_files)}")

    success_count = 0
    no_match_count = 0

    for stt_file in stt_files:
        print(f"\n처리 중: {stt_file.name[:50]}...")

        # 매칭되는 OCR 파일 찾기
        ocr_file = find_matching_ocr_file(stt_file.name, ocr_files)

        if not ocr_file:
            print(f"  ⚠️ OCR 파일 없음")
            no_match_count += 1
            continue

        # 파일 읽기
        with open(stt_file, 'r', encoding='utf-8') as f:
            stt_data = json.load(f)

        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        # 합치기
        merged = merge_stt_ocr(stt_data, ocr_data)

        # 저장
        output_file = OUTPUT_DIR / stt_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        # 통계
        total_segments = len(merged['segments'])
        segments_with_ocr = sum(1 for s in merged['segments'] if 'screen_text' in s)
        print(f"  ✓ 저장: {output_file.name}")
        print(f"    총 세그먼트: {total_segments}, OCR 포함: {segments_with_ocr}")

        success_count += 1

    print(f"\n{'='*60}")
    print(f"완료! 성공: {success_count}, OCR 없음: {no_match_count}")


if __name__ == "__main__":
    main()
