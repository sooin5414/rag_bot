# knowledge_extractor.py
import json
import glob
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
import unicodedata
import re
import difflib

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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


def extract_knowledge_structure(video_data, video_metadata):
    """영상 하나에서 지식 구조 추출"""
    
    # 전체 transcript 합치기
    full_transcript = "\n".join([
        f"[{seg['start']:.1f}s] {seg['text']}" 
        for seg in video_data.get('segments', [])
    ])
    
    video_title = video_metadata.get('title', 'Unknown')
    
    prompt = f"""
            다음은 "{video_title}" 영어 강의의 전체 스크립트입니다.

            {full_transcript[:3000]}  # 너무 길면 truncate

            이 강의의 지식 구조를 JSON으로 추출하세요:

            {{
            "main_topic": "핵심 주제 (예: be동사, 현재완료, 동명사)",
            "sub_topics": [
                {{
                "id": "고유ID (snake_case, 예: be_adjective)",
                "title": "서브토픽 제목 (예: be동사 + 형용사)",
                "concept": "핵심 개념을 1-2문장으로",
                "examples": ["예문1", "예문2"],
                "video_segments": [
                    {{
                    "start_time": 28.5,
                    "end_time": 46.0,
                    "description": "이 구간에서 다루는 내용"
                    }}
                ]
                }}
            ],
            "related_topics": ["이 주제와 연관된 다른 문법 주제"]
            }}

            CRITICAL: 반드시 유효한 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
            """
    
    response = llm.invoke(prompt)
    
    # JSON 파싱
    try:
        # 마크다운 코드블록 제거
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        print(f"응답: {response.content[:200]}")
        return None

def build_knowledge_graph():
    """모든 영상에서 지식 그래프 생성"""
    
    json_files = glob.glob("/data/edutem/sooine/rag_bot/youtube_playlist/*.json")
    
    all_knowledge = {}
    
    print(f"📚 총 {len(json_files)}개 영상 처리 중...")
    
    for i, json_file in enumerate(json_files, 1):
        filename = Path(json_file).name
        print(f"\n[{i}/{len(json_files)}] {filename}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            
            # 비디오 URL 찾기 (유사도 매칭)
            video_url = get_video_url(filename)
            
            if video_url == "URL_없음":
                print(f"  ⚠️  URL을 찾을 수 없음: {filename}")
            
            # 메타데이터 생성
            video_metadata = {
                'title': filename.replace('.json', ''),
                'filename': filename,
                'video_url': video_url
            }
            
            # 지식 구조 추출
            knowledge = extract_knowledge_structure(video_data, video_metadata)
            
            if not knowledge:
                print(f"  ⚠️  지식 추출 실패")
                continue
            
            main_topic = knowledge['main_topic']
            print(f"  ✅ 주제: {main_topic}")
            print(f"     서브토픽 {len(knowledge['sub_topics'])}개 발견")
            print(f"     URL: {video_url}")
            
            # 지식 베이스에 추가
            if main_topic not in all_knowledge:
                all_knowledge[main_topic] = {
                    "definition": "",
                    "sub_topics": {},
                    "videos": []
                }
            
            # 서브토픽 병합
            for sub in knowledge['sub_topics']:
                sub_id = sub['id']
                if sub_id not in all_knowledge[main_topic]['sub_topics']:
                    all_knowledge[main_topic]['sub_topics'][sub_id] = {
                        "title": sub['title'],
                        "concept": sub['concept'],
                        "examples": sub['examples'],
                        "video_segments": []
                    }
                
                # 영상 세그먼트 추가
                for seg in sub['video_segments']:
                    all_knowledge[main_topic]['sub_topics'][sub_id]['video_segments'].append({
                        "video_url": video_url,
                        "video_title": video_metadata['title'],
                        "filename": filename,
                        "start_time": seg['start_time'],
                        "end_time": seg['end_time'],
                        "description": seg['description']
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
    
    # 저장
    output_path = "/data/edutem/sooine/rag_bot/knowledge_graph.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료! 저장 위치: {output_path}")
    print(f"\n📊 통계:")
    print(f"  - 총 주제: {len(knowledge_graph)}개")
    for topic, data in knowledge_graph.items():
        print(f"  - {topic}: 서브토픽 {len(data['sub_topics'])}개, 영상 {len(data['videos'])}개")