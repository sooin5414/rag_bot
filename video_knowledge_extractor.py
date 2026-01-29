
import json
import glob
from pathlib import Path
from dotenv import load_dotenv
import unicodedata
import re
import difflib
from openai import OpenAI

load_dotenv()
client = OpenAI()

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

def normalize_filename(name):
    """파일명 정규화"""
    # NFC 정규화 (맥/리눅스 파일시스템 호환)
    name = unicodedata.normalize("NFC", name)
    # 공백과 특수문자 통일
    name = name.replace("｜", "|").replace("＃", "#")
    name = re.sub(r"\s+", " ", name.strip())
    return name

def get_video_url(video_filename):
    """파일명으로 비디오 URL 찾기 (유사도 매칭)"""
    normalized_filename = normalize_filename(video_filename)

    # 정규화된 키 목록
    normalized_keys = {normalize_filename(k): k for k in video_url_map.keys()}

    # 완전 일치 확인
    if normalized_filename in normalized_keys:
        original_key = normalized_keys[normalized_filename]
        return video_url_map[original_key]

    # 유사도 매칭
    keys = list(normalized_keys.keys())
    match = difflib.get_close_matches(normalized_filename, keys, n=1, cutoff=0.6)

    if match:
        original_key = normalized_keys[match[0]]
        return video_url_map[original_key]

    return "URL_없음"


# ============================================================
# Step 1: 오인식 패턴 추출 (LLM 사용)
# ============================================================
def extract_corrections_from_transcript(transcript_text: str) -> dict:
    """Whisper 오인식 패턴 추출"""
    all_corrections = {}

    chunk_size = 1500
    for i in range(0, len(transcript_text), chunk_size):
        chunk = transcript_text[i:i + chunk_size]
        prompt = f"""당신은 영어 강의 음성인식 오류를 분석하는 전문가입니다.

아래는 한국인 영어 선생님의 강의를 Whisper로 음성인식한 결과입니다.
선생님이 영어 단어를 발음했는데 한글로 잘못 인식된 부분을 찾아주세요.

예시:
 - "be동사" → "비동사"
 - "have" → "해브", "해부"
 - "was" → "워즈"
 - "been" → "빈"

잘못된 예시 (이건 하지 마세요):
- "사람들이" → "people"  (이건 번역임)
- "오늘" → "today" (이건 번역임)

트랜스크립트:
{chunk}

위 텍스트에서 한글로 잘못 인식된 영어 단어들을 찾아서 JSON으로 출력하세요.
{{"한글오인식": "올바른영어", ...}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=800
            )
            corrections = json.loads(response.choices[0].message.content)
            all_corrections.update(corrections)
        except:
            continue

    return all_corrections


def apply_corrections(text: str, corrections: dict) -> str:
    """오인식 패턴을 올바른 영어로 교체"""
    sorted_corrections = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
    for wrong, right in sorted_corrections:
        text = text.replace(wrong, right)
    return text


def apply_corrections_to_segments(segments: list, corrections: dict) -> list:
    """각 segment의 text에 보정 적용"""
    corrected_segments = []
    for seg in segments:
        corrected_seg = seg.copy()
        corrected_seg["text"] = apply_corrections(seg["text"], corrections)
        corrected_segments.append(corrected_seg)
    return corrected_segments


def extract_knowledge_structure(video_data, video_metadata, corrections: dict = None):
    """영상 하나에서 지식 구조 추출"""

    segments = video_data.get('segments', [])

    # 보정 적용
    if corrections:
        segments = apply_corrections_to_segments(segments, corrections)

    # 전체 transcript 합치기 (시간 정보 포함)
    full_transcript = "\n".join([
        f"[{seg['start']:.1f}s ~ {seg['end']:.1f}s] {seg['text']}"
        for seg in segments
    ])

    # 8000자까지 사용 (기존 3000 → 8000)
    if len(full_transcript) > 8000:
        full_transcript = full_transcript[:8000]

    video_title = video_metadata.get('title', 'Unknown')

    prompt = f"""당신은 영어 강의를 분석하는 전문가입니다.
다음은 "{video_title}" 영어 강의의 전체 스크립트입니다.

{full_transcript}

이 강의에서 다루는 모든 문법 주제와 세부 내용을 분석하세요.
하나의 영상에서 여러 sub_topic이 나올 수 있습니다 (예: 기본 설명, 의문문, 부정문, 예문 연습 등).

JSON 형식으로 출력하세요:
{{
  "main_topic": "이 영상의 핵심 문법 주제 (예: 현재진행형, be동사, that절)",
  "definition": "이 문법이 무엇인지 명확하게 요약 (예: '수동태는 주어가 동작을 당하는 것을 표현하는 문법이다. be + 과거분사 형태로 만든다.')",
  "teacher_tip": "선생님이 이 문법을 쉽게 이해시키기 위해 사용한 비유나 핵심 설명. 반드시 문법 개념과 연결된 내용이어야 함. (예: 수동태 - '주어 입장에서 당하는 거예요. I wear a watch면 시계 입장에서는 The watch is worn이 되는 거죠')",
  "sub_topics": [
    {{
      "id": "고유ID (snake_case, 예: present_continuous_question)",
      "title": "서브토픽 제목 (예: 현재진행형 의문문)",
      "concept": "이 서브토픽의 핵심 개념 요약 (1-2문장)",
      "teacher_explanation": "선생님이 이 부분을 설명할 때 핵심 한마디 (짧고 이해하기 쉬운 표현)",
      "examples": ["I have studied for two years. (나는 2년 동안 공부해왔어)", "We have met before. (우리 전에 만난 적 있어)"],
      "video_segments": [
        {{
          "start_time": 28.5,
          "end_time": 46.0,
          "description": "이 구간에서 다루는 내용 요약"
        }}
      ]
    }}
  ],
  "related_topics": ["연관 문법 주제1", "연관 문법 주제2"]
}}

중요:
- definition은 문법 개념을 명확하게 정의하세요 (스크립트 복사 X, 요약 O)
- teacher_tip은 반드시 문법 개념을 설명하는 비유/팁이어야 함 (단순 문장 예시 X)
  - 좋은 예: "수동태는 주어가 동작을 당하는 입장이에요"
  - 나쁜 예: "내가 시계를 차면, 시계 입장에서는 차지는 거예요" (이건 그냥 예시문)
- teacher_explanation도 문법 개념 설명이어야 함 (1-2문장)
- sub_topics는 최소 2개 이상 추출하세요
- video_segments의 시간은 스크립트의 [시간] 정보를 참고하세요
- examples는 실제 강의에서 나온 완전한 영어 문장을 사용하세요
- 각 예문은 반드시 "영어 문장. (한국어 번역)" 형식으로 작성하세요
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ API 에러: {e}")
        return None

def build_knowledge_graph():
    """모든 영상에서 지식 그래프 생성"""

    json_files = glob.glob("/data/edutem/sooine/rag_bot/merged_data/*.json")

    all_knowledge = {}

    print(f"📚 총 {len(json_files)}개 영상 처리 중...")

    for i, json_file in enumerate(json_files, 1):
        filename = Path(json_file).name
        print(f"\n[{i}/{len(json_files)}] {filename[:50]}...")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                video_data = json.load(f)

            # 비디오 URL 찾기 (유사도 매칭)
            video_url = get_video_url(filename)

            if video_url == "URL_없음":
                print(f"  ⚠️  URL을 찾을 수 없음")

            # Step 1: 오인식 보정
            segments = video_data.get('segments', [])
            # STT + OCR 텍스트 합치기
            transcript_text = " ".join([
                seg.get("text", "") + (" [화면: " + " ".join(seg.get("screen_text", [])) + "]" if seg.get("screen_text") else "")
                for seg in segments
            ])
            print(f"  🔧 오인식 보정 중...")
            corrections = extract_corrections_from_transcript(transcript_text)
            if corrections:
                print(f"     {len(corrections)}개 패턴 발견")

            # 메타데이터 생성
            video_metadata = {
                'title': filename.replace('.json', ''),
                'filename': filename,
                'video_url': video_url
            }

            # Step 2: 지식 구조 추출 (보정된 텍스트로)
            print(f"  📖 지식 구조 추출 중...")
            knowledge = extract_knowledge_structure(video_data, video_metadata, corrections)

            if not knowledge:
                print(f"  ⚠️  지식 추출 실패")
                continue

            main_topic = knowledge.get('main_topic', 'Unknown')
            definition = knowledge.get('definition', '')
            teacher_tip = knowledge.get('teacher_tip', '')
            sub_topics = knowledge.get('sub_topics', [])

            print(f"  ✅ 주제: {main_topic}")
            print(f"     서브토픽 {len(sub_topics)}개 발견")

            related_topics = knowledge.get('related_topics', [])

            # 지식 베이스에 추가
            if main_topic not in all_knowledge:
                all_knowledge[main_topic] = {
                    "definition": definition,
                    "teacher_tip": teacher_tip,
                    "sub_topics": {},
                    "videos": [],
                    "related_topics": set()
                }
            else:
                # 비어있으면 업데이트
                if not all_knowledge[main_topic]["definition"] and definition:
                    all_knowledge[main_topic]["definition"] = definition
                if not all_knowledge[main_topic]["teacher_tip"] and teacher_tip:
                    all_knowledge[main_topic]["teacher_tip"] = teacher_tip

            # related_topics 병합
            all_knowledge[main_topic]["related_topics"].update(related_topics)

            # 서브토픽 병합
            for sub in sub_topics:
                sub_id = sub.get('id', 'unknown')
                if sub_id not in all_knowledge[main_topic]['sub_topics']:
                    all_knowledge[main_topic]['sub_topics'][sub_id] = {
                        "title": sub.get('title', ''),
                        "concept": sub.get('concept', ''),
                        "teacher_explanation": sub.get('teacher_explanation', ''),
                        "examples": sub.get('examples', []),
                        "video_segments": []
                    }

                # 영상 세그먼트 추가
                for seg in sub.get('video_segments', []):
                    all_knowledge[main_topic]['sub_topics'][sub_id]['video_segments'].append({
                        "video_url": video_url,
                        "video_title": video_metadata['title'],
                        "filename": filename,
                        "start_time": seg.get('start_time', 0),
                        "end_time": seg.get('end_time', 0),
                        "description": seg.get('description', '')
                    })

            # 비디오 메타데이터 추가
            all_knowledge[main_topic]['videos'].append({
                "url": video_url,
                "title": video_metadata['title'],
                "filename": filename
            })

        except Exception as e:
            print(f"  ❌ 에러: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_knowledge

if __name__ == "__main__":
    print("🚀 지식 그래프 생성 시작...\n")

    knowledge_graph = build_knowledge_graph()

    # set을 list로 변환 (JSON 저장용)
    for topic in knowledge_graph:
        if isinstance(knowledge_graph[topic].get("related_topics"), set):
            knowledge_graph[topic]["related_topics"] = list(knowledge_graph[topic]["related_topics"])

    # 저장
    output_path = "/data/edutem/sooine/rag_bot/knowledge_graph.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료! 저장 위치: {output_path}")
    print(f"\n📊 통계:")
    print(f"  - 총 주제: {len(knowledge_graph)}개")
    for topic, data in knowledge_graph.items():
        print(f"  - {topic}: 서브토픽 {len(data['sub_topics'])}개, 영상 {len(data['videos'])}개")