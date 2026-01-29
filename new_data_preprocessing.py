from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import json
import glob
import re
import unicodedata
import difflib

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
    """맥OS/리눅스 호환을 위한 파일명 정규화"""
    name = unicodedata.normalize("NFC", name)
    name = name.replace("｜", "|").replace("＃", "#")
    name = re.sub(r"\s+", " ", name.strip())
    return name


def get_video_url(filename):
    """
    Whisper로 생성된 JSON 파일명과 video_url_map 키가 다를 수 있음.
    fuzzy matching으로 가장 비슷한 key를 찾아 URL 매핑.
    """
    normalized_keys = list(video_url_map.keys())
    match = difflib.get_close_matches(filename, normalized_keys, n=1, cutoff=0.6)
    if match:
        return video_url_map[match[0]]
    return "URL_없음"

# Step 1: 오인식 패턴 추출 (LLM 사용)
# ============================================================
def extract_corrections_from_transcript(transcript_text: str) -> dict:
    all_corrections = {}
    
    # 1500자씩 청크로 나눔
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
 - "been" → "빈", "빈"
 
 잘못된 예시 (이건 하지 마세요): 영어단어를 번역은 하지마세요
- "사람들이" → "people"  (이건 번역임)
- "오늘" → "today" (이건 번역임)

트랜스크립트:
{transcript_text[:2000]}

위 텍스트에서 한글로 잘못 인식된 영어 단어들을 찾아서 JSON으로 출력하세요.
반드시 아래 형식으로만 출력하세요.

{{"한글오인식": "올바른영어", ...}}

예시: {{"워킹": "working", "스튜디": "studying", "비": "be"}}
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
            all_corrections.update(corrections)  # 합침
        except:
            continue
    
    return all_corrections


# ============================================================
# Step 2: 텍스트 보정 적용
# ============================================================
def apply_corrections(text: str, corrections: dict) -> str:
    """
    오인식 패턴을 올바른 영어로 교체
    주의: 긴 패턴부터 먼저 교체 (부분 매칭 방지)
    """
    # 긴 패턴부터 교체 (예: "비 워킹" → "be working" 먼저)
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


# ============================================================
# Step 3: LLM Chunking (주제별 구간 병합)
# ============================================================
def chunk_segments_with_llm(segments: list, video_title: str) -> dict:
    transcript_block = ""
    for seg in segments:
        transcript_block += f"[{seg['start']:.1f} ~ {seg['end']:.1f}] {seg['text'].strip()}\n"

    if len(transcript_block) > 8000:
        transcript_block = transcript_block[:8000]

    prompt = f"""
당신은 영어 강의를 구조화하는 교육 콘텐츠 분석가입니다.

아래는 '{video_title}'의 Whisper 전사본입니다.

📌 각 chunk에 대해 **교육적 역할(role)**과 **문법 분류(grammar_type)**를 판단하세요.

role은 아래 4개 중 하나:
- definition   : 개념이 무엇인지 설명
- usage        : 문법을 어떻게 쓰는지 설명
- comparison   : 다른 문법과 비교
- practice     : 연습문제 / 예문 풀이

grammar_type은 아래 분류 기준을 따르세요:
- 시제: 과거/현재/미래/완료/진행형 등
- be동사: am/is/are/was/were 단독 설명 (조동사 아님!)
- 조동사: can/will/would/should/must/may 단독 설명 (be동사 제외!)
- 문장구조: that절/관계대명사/수동태/to부정사/동명사/조동사+be동사 결합(can be 등) 등
- 전치사: in/on/at/for/with 등
- 동사: 일반 동사 활용
- 기타: 위에 해당 안 됨

⚠️ 중요:
- be동사(am/is/are)는 조동사가 아님!
- 조동사는 can/will/should 등만 해당
- 조동사+be동사 결합(can be, will be 등)은 "문장구조"로 분류!

출력 형식(JSON only):
{{
  "chunks": [
    {{
      "topic": "영어 문법 용어",
      "role": "definition | usage | comparison | practice",
      "summary": "50자 이내 요약",
      "grammar_type": "시제 | be동사 | 조동사 | 문장구조 | 전치사 | 동사 | 기타",
      "start_time": 0.0,
      "end_time": 0.0,
      "content": "200자 이내 핵심 설명",
      "keywords": ["키워드1", "키워드2"]
    }}
  ]
}}

Whisper 전사본:
{transcript_block}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=8000
    )
    return json.loads(response.choices[0].message.content)



# ============================================================
# Step 4: Document 생성
# ============================================================
def build_documents_from_chunks(chunked, video_url, video_filename):
    chunks = chunked.get("chunks", [])
    docs = []

    for chunk in chunks:
        keywords = chunk.get("keywords", [])
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else ""

        doc = Document(
            page_content=chunk["content"],
            metadata={
                "topic": chunk["topic"],
                "role": chunk["role"],   # ✅ 핵심 추가
                "summary": chunk["summary"],
                "grammar_type": chunk.get("grammar_type", ""),
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "video_url": video_url,
                "video_filename": video_filename,
                "keywords": keywords_str,
            }
        )
        docs.append(doc)

    return docs


import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

json_files = sorted(glob.glob("./merged_data/*.json"))

all_docs = []

for json_file in json_files:
    video_filename = normalize_filename(os.path.basename(json_file))
    video_url = get_video_url(video_filename)
    
    with open(json_file, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    segments = result["segments"]
    # STT + OCR 텍스트 합치기
    transcript_text = " ".join([
        seg["text"] + (" [화면: " + " ".join(seg.get("screen_text", [])) + "]" if seg.get("screen_text") else "")
        for seg in segments
    ])

    corrections = extract_corrections_from_transcript(transcript_text)
    corrected_segments = apply_corrections_to_segments(segments, corrections)
    chunked = chunk_segments_with_llm(corrected_segments, video_filename)
    docs = build_documents_from_chunks(chunked, video_url, video_filename)
    
    all_docs.extend(docs)  # 누적
    print(f"✅ {video_filename}: {len(docs)}개 docs")

# 전부 처리 후 저장
print(f"\n총 {len(all_docs)}개 documents")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory="./chroma_db_with_role"
)
print("✅ 저장 완료!")
