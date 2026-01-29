"""
=====================================================
🔥 전략 B: LLM 기반 의미 단위 Chunking 전체 파이프라인
Whisper Segment → LLM Chunker → Document 병합 → Vectorstore 저장

작성자: GPT
설명: Whisper로 쪼개진 3~7초짜리 segment들을
      LLM이 문법 주제별 ‘의미 단위 섹션(chunk)’ 으로 자동 그룹화한다.

이 Chunk 단위를 Vectorstore에 넣어야
"동명사 부분만 틀어줘", "현재완료 설명 찾아줘"가 정확하게 가능하다.
=====================================================
"""

import json
import glob
import os
import unicodedata
import re
import difflib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =====================================================
# 0. 환경 변수 로드
# =====================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# =====================================================
# 1. YouTube JSON 파일명 → 영상 URL 매핑 테이블
# =====================================================

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

# =====================================================
# 2. 파일명 정규화 + URL 찾기 (fuzzy match)
# =====================================================
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



# =====================================================
# 3. LLM Chunker (핵심)
# =====================================================
def chunk_segments_with_llm(segments, video_title):
    """
    반환 형식:
    {{
      "chunks": [
        {{
          "topic": "...",
          "summary": "...",
          "start_time": 123.4,
          "end_time": 175.9,
          "content": "해당 구간 전체 텍스트"
        }},
        ...
      ]
    }}
    """


    # Whisper segment 전체를 하나의 문자열로 합침
    transcript_block = ""
    for seg in segments:
        s = seg["start"]
        e = seg["end"]
        t = seg["text"].strip()
        transcript_block += f"[{s:.1f} ~ {e:.1f}] {t}\n"

    print(f"🤖 LLM Chunking: {video_title} ...")

    prompt = fprompt = f"""
          당신은 영어 강의를 전문적으로 구조화하는 분석가입니다.
          아래는 '{video_title}' 의 전체 Whisper 전사본입니다.

          Whisper segment들은 짧게 쪼개져 있으니,
          이를 '문법 주제별 의미 단위(chunk)'로 자동으로 묶어주세요.

          Chunk 하나는 반드시 다음을 포함해야 합니다:

          1) topic       : 이 chunk의 핵심 문법 주제 (예: "현재완료", "동명사", "that 용법")
          2) summary     : chunk의 핵심을 1~2문장 요약
          3) start_time  : 이 chunk가 시작되는 Whisper segment의 첫 start_time
          4) end_time    : 이 chunk가 끝나는 Whisper segment의 마지막 end_time
          5) content     : chunk에 포함된 segment들의 텍스트를 모두 합친 것

          출력 형식(JSON only):

          {{
            "chunks": [
              {{
                "topic": "동명사",
                "summary": "동명사가 문장에서 명사처럼 쓰이는 원리를 설명한다.",
                "start_time": 12.3,
                "end_time": 45.8,
                "content": "...."
              }}
            ]
          }}

          아래는 Whisper 전체 전사본입니다:

          {transcript_block}
"""


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)



# =====================================================
# 4. 문서를 Vectorstore에 넣기 위한 형태로 변환
# =====================================================
def build_documents_from_chunks(chunks, video_url, video_filename):
    """
    LLM이 만든 chunk들을 LangChain Document 형태로 변환
    """
    documents = []

    for ch in chunks:
        doc = Document(
            page_content=ch["content"],   # 검색용 전체 텍스트
            metadata={
                "topic": ch.get("topic", ""),
                "summary": ch.get("summary", ""),
                "video_url": video_url,
                "video_filename": video_filename,
                "start_time": ch.get("start_time", 0),
                "end_time": ch.get("end_time", 0),
            }
        )
        documents.append(doc)

    return documents



# =====================================================
# 5. 전체 파일 처리 → Chunk → Documents 변환
# =====================================================
def process_all_videos():
    """
    youtube_playlist/*.json 폴더에서 파일 가져와
    Whisper segments → LLM Chunking → Documents 생성
    """
    json_files = sorted(glob.glob("./youtube_playlist/*.json"))

    all_docs = []
    print(f"📄 총 {len(json_files)}개 파일 처리 시작\n")

    for json_file in json_files:
        file_name = Path(json_file).name
        video_url = get_video_url(file_name)

        print(f"🎬 파일 처리: {file_name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data["segments"]

        # ⚠ Whisper segments → LLM으로 섹션 자동 병합
        chunked = chunk_segments_with_llm(segments, file_name)

        # 각 chunk → LangChain Document로 변환
        docs = build_documents_from_chunks(
            chunks=chunked["chunks"],
            video_url=video_url,
            video_filename=file_name
        )

        all_docs.extend(docs)

    print(f"\n✅ 전체 문서 생성 완료: {len(all_docs)}개\n")
    return all_docs



# =====================================================
# 6. Vectorstore 생성 및 저장
# =====================================================
def save_vectorstore(documents):
    print("🔧 Embedding 생성 중...")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("💽 Vectorstore 생성 및 저장...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="english_grammar_chunked"
    )

    print("🎉 Vectorstore 저장 완료!")



# =====================================================
# 7. 실행
# =====================================================
if __name__ == "__main__":
    docs = process_all_videos()
    save_vectorstore(docs)
