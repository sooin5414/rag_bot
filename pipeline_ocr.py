import cv2
from paddleocr import PaddleOCR
import json
from pathlib import Path
import subprocess
import numpy as np

# ========== 설정 ==========
CROP_REGION = (0.02, 0.12, 0.80, 0.90)  # 변화 감지 및 OCR 영역
THRESHOLD = 5  # 화면 변화 감지 임계값
MIN_INTERVAL = 2  # 최소 프레임 간격(초)
OUTPUT_DIR = Path("ocr_results")  # 결과 저장 폴더

# video_knowledge_extractor.py에서 가져온 URL 매핑
video_url_map = {
    "01_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #29 | 과거에 있었던 일 설명할 때  | 기초 영어 회화.json": "https://www.youtube.com/watch?v=R_-pgaQYaYQ",
    "02_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #28 | 마법과 같은 that | 기초 영어 회화.json": "https://www.youtube.com/watch?v=008886a-lQI",
    "03_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 ##27 | 수동태를 that과 연결하기! | 기초 영어 회화.json": "https://www.youtube.com/watch?v=r3qBF9dMz10",
    "04_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #26 | 수동태 핵심 파악! | 기초 영어 회화.json": "https://www.youtube.com/watch?v=bGl_7acUnNk",
    "05_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #25 | have를 깊이, 자연스럽게 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=jOb8mznvX48",
    "06_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #24 | 현재완료 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=RgNbTRRt78Y",
    "07_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #23 | that 뒤에 조동사 쓰기  | 기초 영어 회화.json": "https://www.youtube.com/watch?v=OaTujaboBf8",
    "08_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #22 | that 뒤에 진행형을 가지고 만들어 보자  | 기초 영어 회화.json": "https://www.youtube.com/watch?v=NXbGg9nxpdk",
    "09_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #21 | 또 다른 that | 기초 영어 회화.json": "https://www.youtube.com/watch?v=k3-666q27Ps",
    "10_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #20 | that으로 문장을 길게! (3) | 기초 영어 회화.json": "https://www.youtube.com/watch?v=JMfB_2pfqCA",
    "11_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #19 | that 으로 문장 길게 만들기(2) | 기초 영어 회화.json": "https://www.youtube.com/watch?v=gHrI6qhbziI",
    "12_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#18 | that 으로 문장 길게 만들기(1) | 기초 영어 회화.json": "https://www.youtube.com/watch?v=nHbEN7KEmmE",
    "13_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #17 | can을 be able to로 바꾸기! | 기초 영어 회화.json": "https://www.youtube.com/watch?v=CGN1TdvhkvY",
    "14_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #16 | 동명사로 주어 길게 만들기 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=jzJzdoBdeAc",
    "15_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #15 | 목적을 나타내는 to 동사원형 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=7CvXgPmdD9s",
    "16_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #14 | 미래와 과거에도 진행형을 쓴다! | 기초 영어 회화.json": "https://www.youtube.com/watch?v=7ot7hY8wm4Q",
    "17_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #13 | 현재 진행형의 다양한 쓰임.json": "https://www.youtube.com/watch?v=j87QB9EZZrY",
    "18_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #12 | '~에 있다'를 뜻하는 be 동사 연습 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=h_Yv5bX8p8k",
    "19_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#11 | 다양한 be 동사 형태 연습 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=WS5hLZV7Lb4",
    "20_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#10 | 바쁘다는 busy가 아니다! | 기초 영어 회화.json": "https://www.youtube.com/watch?v=V8e_cwY7VTs",
    "21_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#9 | if로 문장 길게 만들기 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=hh9pAAS-gho",
    "22_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#8 | '~하기를' to 동사원형 연습 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=TK1HL_27g6U",
    "23_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #5  | 의문문 길게 만들기 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=EXKS6rZHbbA",
    "24_NEW 이시원의 기초 영어 회화 강의한 달 만에 영어로 말문 트기 #4 | 과거 시제 마스터 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=G_SNroMhJTQ",
    "25_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#3 | And 로 문장을 길게!  | 기초 영어 회화.json": "https://www.youtube.com/watch?v=VJeidy58uJQ",
    "26_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#1 | 영어는 단어의 연결 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=oLIpoVoDgTo",
    "27_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#2 | 미래를 나타내는 will | 기초 영어 회화.json": "https://www.youtube.com/watch?v=KQbWy6j_TFA",
    "28_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #30 | 많이 쓰는 동사 put / get / take | 기초 영어 회화.json": "https://www.youtube.com/watch?v=1dIALFMvJlA",
    "29_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기#6 | 의문문 질문에 답하기 | 기초 영어 회화.json": "https://www.youtube.com/watch?v=a_eNZ4ZVwxc",
    "30_NEW 이시원 강의 | 한 달 만에 영어로 말문 트기 #7 | 가능과 허락의 can | 기초 영어 회화.json": "https://www.youtube.com/watch?v=kYx8f4U4-jo"
}

# PaddleOCR 초기화 (한국어)
print("PaddleOCR 초기화 중...")
ocr = PaddleOCR(lang='korean')
print("PaddleOCR 초기화 완료")


def get_stream_url(youtube_url):
    """YouTube URL에서 스트림 URL 추출 (MP4 우선)"""
    yt_dlp_path = '/data/edutem/sooine/paddleocr_env/bin/yt-dlp'
    # HLS 대신 MP4 포맷 우선 선택
    cmd = [yt_dlp_path, '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', '-g', youtube_url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    urls = result.stdout.strip().split('\n')
    # 첫 번째 URL (비디오) 반환
    return urls[0] if urls else ''


def extract_changed_frames(video_url, threshold=5, min_interval=2, crop_region=None):
    """화면이 바뀔 때만 프레임 추출"""
    stream_url = get_stream_url(video_url)
    if not stream_url:
        print(f"  스트림 URL 추출 실패: {video_url}")
        return []

    cap = cv2.VideoCapture(stream_url)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    prev_gray = None
    frame_count = 0
    last_saved_time = -min_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if crop_region:
            h, w = gray.shape
            x1 = int(w * crop_region[0])
            y1 = int(h * crop_region[1])
            x2 = int(w * crop_region[2])
            y2 = int(h * crop_region[3])
            gray_crop = gray[y1:y2, x1:x2]
        else:
            gray_crop = gray

        should_save = False

        if prev_gray is None:
            should_save = True
        else:
            diff = cv2.absdiff(prev_gray, gray_crop)
            change = np.mean(diff)
            if change > threshold and (timestamp - last_saved_time) >= min_interval:
                should_save = True

        if should_save:
            frames.append({
                "timestamp": round(timestamp, 2),
                "frame": frame
            })
            last_saved_time = timestamp

        prev_gray = gray_crop
        frame_count += 1

    cap.release()
    return frames


def extract_text_from_frames(frames, ocr_crop=None):
    """프레임에서 OCR로 텍스트 추출"""
    results = []
    for i, item in enumerate(frames):
        frame = item["frame"]
        timestamp = item["timestamp"]

        if ocr_crop:
            h, w = frame.shape[:2]
            x1 = int(w * ocr_crop[0])
            y1 = int(h * ocr_crop[1])
            x2 = int(w * ocr_crop[2])
            y2 = int(h * ocr_crop[3])
            frame_crop = frame[y1:y2, x1:x2]
        else:
            frame_crop = frame

        ocr_result = ocr.predict(frame_crop)

        texts = []
        if isinstance(ocr_result, list) and len(ocr_result) > 0:
            result_dict = ocr_result[0]
            if isinstance(result_dict, dict):
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                for text, score in zip(rec_texts, rec_scores):
                    if score > 0.5 and text.strip():
                        texts.append(text)
        elif isinstance(ocr_result, dict):
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            for text, score in zip(rec_texts, rec_scores):
                if score > 0.5 and text.strip():
                    texts.append(text)

        if texts:
            results.append({
                "timestamp": round(timestamp, 2),
                "texts": texts
            })

    return results


def process_video(video_name, video_url):
    """단일 영상 처리"""
    print(f"\n{'='*60}")
    print(f"처리 중: {video_name[:50]}...")
    print(f"URL: {video_url}")

    # 프레임 추출
    frames = extract_changed_frames(video_url, THRESHOLD, MIN_INTERVAL, CROP_REGION)
    print(f"  {len(frames)}개 프레임 추출")

    if not frames:
        return None

    # OCR 처리
    ocr_results = extract_text_from_frames(frames, CROP_REGION)
    print(f"  {len(ocr_results)}개 OCR 결과")

    return {
        "video_name": video_name,
        "video_url": video_url,
        "ocr_data": ocr_results
    }


def main():
    """전체 영상 처리"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = {}
    total = len(video_url_map)

    for idx, (video_name, video_url) in enumerate(video_url_map.items(), 1):
        print(f"\n[{idx}/{total}] 진행 중...")

        # 이미 처리된 파일 스킵
        output_file = OUTPUT_DIR / f"{video_name.replace('.json', '_ocr.json')}"
        if output_file.exists():
            print(f"  이미 처리됨: {output_file.name}")
            continue

        try:
            result = process_video(video_name, video_url)
            if result:
                # 개별 파일 저장
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"  저장: {output_file.name}")

                all_results[video_name] = result
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue

    # 전체 결과 저장
    if all_results:
        with open(OUTPUT_DIR / "all_ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n전체 결과 저장: {OUTPUT_DIR / 'all_ocr_results.json'}")

    print(f"\n{'='*60}")
    print(f"완료! 총 {len(all_results)}개 영상 처리")


if __name__ == "__main__":
    main()
