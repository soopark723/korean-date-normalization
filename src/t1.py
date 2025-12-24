import json
import os
import time
import argparse
import re
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from openai import OpenAI
from tqdm import tqdm


KASI_API_KEY = os.getenv("KASI_API_KEY", "PUT YOUR API KEY HERE") 

def execute_calculator(tool_input: str) -> str:
    """
    날짜 계산 도구. '2025-11-21 + 7 days', '2025-11-21 next friday', '2025-11-21 next month' 같은 다양한 날짜 계산 입력을 처리합니다.
    """
    try:
        tool_input = tool_input.lower().strip()

        # 패턴 1: 'YYYY-MM-DD +/- N unit' 형식 (e.g., 2025-11-21 + 3 weeks)
        pattern1 = r"(\d{4}-\d{2}-\d{2})\s*([+-])\s*(\d+)\s*(days?|weeks?|months?)"
        match1 = re.match(pattern1, tool_input)
        if match1:
            base_date_str, operator, num_str, unit = match1.groups()
            base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
            num = int(num_str)

            if unit.startswith("day"):
                delta = timedelta(days=num)
            elif unit.startswith("week"):
                delta = timedelta(weeks=num)
            elif unit.startswith("month"):
                delta = relativedelta(months=num)
            
            result_date = base_date + delta if operator == '+' else base_date - delta
            return result_date.strftime("%Y-%m-%d")

        # 패턴 2: 'YYYY-MM-DD [next/last/previous/this] weekday' 형식 (e.g., 2025-11-21 next friday)
        pattern2 = r"(\d{4}-\d{2}-\d{2})\s*(next|last|previous|this)\s*(\w+day)"
        match2 = re.match(pattern2, tool_input)
        if match2:
            base_date_str, direction, day_name = match2.groups()
            base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
            
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if day_name not in weekdays:
                return f"Error: Unknown day '{day_name}'"
            
            target_weekday = weekdays.index(day_name)
            current_weekday = base_date.weekday()
            
            if direction in ["next", "this"]:
                days_ahead = target_weekday - current_weekday
                if direction == "next" or (direction == "this" and days_ahead < 0):
                     days_ahead += 7
                result_date = base_date + timedelta(days_ahead)
            elif direction in ["last", "previous"]:
                days_behind = current_weekday - target_weekday
                if days_behind <= 0:
                    days_behind += 7
                result_date = base_date - timedelta(days_behind)
            
            return result_date.strftime("%Y-%m-%d")

        # 패턴 3: 'YYYY-MM-DD [next/last/previous/this] week/month' 형식 (e.g., 2025-11-21 next month)
        pattern3 = r"(\d{4}-\d{2}-\d{2})\s*(next|last|previous|this)\s*(week|month)"
        match3 = re.match(pattern3, tool_input)
        if match3:
            base_date_str, direction, unit = match3.groups()
            base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
            
            delta = None
            if unit == "week":
                delta = timedelta(weeks=1)
            elif unit == "month":
                delta = relativedelta(months=1)

            if direction in ["next", "this"]:
                result_date = base_date + delta
            elif direction in ["last", "previous"]:
                result_date = base_date - delta
            
            return result_date.strftime("%Y-%m-%d")

        return f"Error: Cannot parse calculator input '{tool_input}'"

    except Exception as e:
        return f"Calculator Error: {str(e)}"


def execute_calendar_db(tool_input: dict) -> str:
    """
    KASI 특일 정보 API를 호출하여 공휴일, 기념일 등의 정보를 가져옵니다.
    tool_input 예시: {"year": "2025", "month": "all", "category": "rest"}
    """
    if not isinstance(tool_input, dict):
        return "Error: Input for calendar_db must be a dictionary."

    year = tool_input.get("year")
    month = tool_input.get("month")
    category = tool_input.get("category", "rest")
    
    if not year or not month:
        return "Error: 'year' and 'month' are required for calendar_db."

    category_map = {
        "holiday": "getHoliDeInfo",
        "rest": "getRestDeInfo",
        "anniversary": "getAnniversaryInfo",
        "24divisions": "get24DivisionsInfo", 
        "sundry": "getSundryDayInfo" 
    }
    
    operation_name = category_map.get(category, "getRestDeInfo")
    base_url = f"http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/{operation_name}"
    
    date_kind_map = {
        "01": "국경일",
        "02": "기념일",
        "03": "24절기",
        "04": "잡절"
    }

    months_to_query = []
    if str(month).lower() == "all":
        months_to_query = [f"{i:02d}" for i in range(1, 13)]
    elif "," in str(month):
        months_to_query = [m.strip().zfill(2) for m in str(month).split(",")]
    else:
        months_to_query = [str(month).zfill(2)]
        
    all_results = []
    for m in months_to_query:
        params = {
            "solYear": year,
            "solMonth": m,
            "ServiceKey": KASI_API_KEY,
            "_type": "json",
            "numOfRows": 50
        }
        try:
            if KASI_API_KEY == "YOUR_KASI_API_KEY_HERE":
                 raise ValueError("KASI_API_KEY is not set.")
            res = requests.get(base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            items = data.get('response', {}).get('body', {}).get('items', {}).get('item')
            if not items:
                continue
            if isinstance(items, dict): 
                items = [items]
            
            for item in items:
                date_kind_code = item.get('dateKind')
                result_item = {
                    "dateName": item.get('dateName'),
                    "locdate": str(item.get('locdate')),
                    "isHoliday": item.get('isHoliday', 'N'), 
                    "dateKind": date_kind_map.get(date_kind_code, date_kind_code) 
                }
                all_results.append(result_item)
        except Exception as e:
            all_results.append(f"API Error for {year}-{m}: {str(e)}")
            
    return json.dumps(all_results, ensure_ascii=False) if all_results else "No special days found."

def execute_search(tool_input: str) -> str:
    """
    검색 도구. Solar 모델을 사용하여 입력된 쿼리에 대한 정보를 검색하고 요약합니다.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise, factual answers based on the user's query, as if you were a search engine."},
            {"role": "user", "content": f"For the query '{tool_input}', provide a direct and factual answer, summarizing the key information within 150 characters."}
        ]
        
        response = client.chat.completions.create(
            model="solar-pro2",  # adjust model as needed
            messages=messages,
            temperature=0
        )
        
        search_result = response.choices[0].message.content.strip()
        return search_result

    except Exception as e:
        return f"Search tool error: {str(e)}"

# --- 1. 실행 인자 설정 ---
parser = argparse.ArgumentParser(description="Run model with CoT or ReAct method.")
parser.add_argument(
    '--method', 
    type=str, 
    choices=['cot', 'react'], 
    required=True, 
    help="Method to use: 'cot' for Chain-of-Thought, 'react' for ReAct."
)
args = parser.parse_args()

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY", "PUT YOUR API KEY HERE"),
    base_url="https://api.upstage.ai/v1"
)

# --- 2. 메소드에 따라 프롬프트 로드 ---
system_prompt = ""
observation_prompt = ""

if args.method == 'cot':
    prompt_filepath = '/prompts/t1_cot.txt'  # adjust as needed
    try:
        with open(prompt_filepath, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()
elif args.method == 'react':
    thought_prompt_filepath = '/prompts/t1_react_thought.txt'  # adjust as needed
    try:
        with open(thought_prompt_filepath, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{thought_prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()
        
    observation_prompt_filepath = '/prompts/t1_react_observation.txt'  # adjust as needed
    try:
        with open(observation_prompt_filepath, 'r', encoding='utf-8') as f:
            observation_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{observation_prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()

# 3. 데이터셋 불러오기
try:
    with open('/data/T1_dataset.json', 'r', encoding='utf-8') as f:  # adjust as needed
        dataset = json.load(f)
except FileNotFoundError:
    print("오류: '/data/T1_dataset.json' 파일을 찾을 수 없습니다.")  # adjust as needed
    exit()

results = []

# 4. 데이터셋의 각 항목에 대해 반복 작업 수행
for item in tqdm(dataset, desc=f"데이터 처리 중 ({args.method.upper()})"):
    input_text = item.get("input_text")
    anchor_date = item.get("anchor_date")

    if not input_text or not anchor_date:
        item['prediction'] = {"error": "Missing input_text or anchor_date"}
        results.append(item)
        continue

    # --- 5. CoT 와 ReAct 로직 분기 ---
    if args.method == 'cot':
        # --- CoT 로직 ---
        user_input_json = {"input_text": input_text, "anchor_date": anchor_date}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_input_json, ensure_ascii=False, indent=2)}
        ]
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="solar-pro2", messages=messages, temperature=0, response_format={"type": "json_object"}  # adjust model as needed
            )
            latency = time.time() - start_time
            prediction_text = response.choices[0].message.content.strip()
            
            try:
                prediction_json = json.loads(prediction_text)
                item['thought'] = prediction_json.get("thought", "Thought key not found")
                item['prediction'] = prediction_json.get("prediction", "Prediction key not found")
            except json.JSONDecodeError:
                item['prediction'] = f"Error: Invalid JSON response: {prediction_text}"
                item['thought'] = "N/A due to invalid JSON response"

            item['latency'] = latency
            usage = getattr(response, "usage", None)
            item['tokens'] = getattr(usage, "total_tokens", None) if usage else None
        except Exception as e:
            print(f"ID {item.get('id')} 처리 중 오류 발생: {e}")
            item['prediction'] = f"Error: {str(e)}"
        results.append(item)

    elif args.method == 'react':
        # --- ReAct 로직 ---
        start_time = time.time()
        total_tokens = 0
        
        try:
            # [Step 1: Thought & Tool Selection]
            user_input_json = {"input_text": input_text, "anchor_date": anchor_date}
            messages_step1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_input_json, ensure_ascii=False)}
            ]
            response_step1 = client.chat.completions.create(
                model="solar-pro2", messages=messages_step1, temperature=0, response_format={"type": "json_object"}  # adjust model as needed
            )
            total_tokens += getattr(response_step1, "usage", {}).total_tokens or 0
            step1_output = json.loads(response_step1.choices[0].message.content)
            
            tool_name = step1_output.get("tool")
            tool_input = step1_output.get("tool_input")
            thought = step1_output.get("thought")
            item['react_step1_output'] = step1_output

            # [Step 2: Action (Tool Execution)]
            observation = ""
            if tool_name == "calculator":
                observation = execute_calculator(tool_input)
            elif tool_name == "calendar_db":
                observation = execute_calendar_db(tool_input)
            elif tool_name == "search":
                observation = execute_search(tool_input)
            elif tool_name == "finish":
                observation = "No tool needed. Directly providing the answer."
                item['prediction'] = tool_input
            else:
                observation = "Error: Unknown tool selected or tool not provided."
            
            item['react_observation'] = observation

            # [Step 3: Final Answer Generation (if needed)]
            if tool_name != "finish":
                tool_log = {
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": observation
                }
                final_user_input = {
                    "input_text": input_text,
                    "anchor_date": anchor_date,
                    "tool_log": tool_log
                }
                final_user_content = json.dumps(final_user_input, ensure_ascii=False, indent=2)
                
                messages_step3 = [
                    {"role": "system", "content": observation_prompt},
                    {"role": "user", "content": final_user_content}
                ]
                response_step3 = client.chat.completions.create(
                    model="solar-pro2", messages=messages_step3, temperature=0, response_format={"type": "json_object"}  # adjust model as needed
                )
                total_tokens += getattr(response_step3, "usage", {}).total_tokens or 0
                step3_output = json.loads(response_step3.choices[0].message.content)

                item['thought'] = step3_output.get("thought")
                item['prediction'] = step3_output.get("prediction")

            else:
                item['thought'] = thought

            latency = time.time() - start_time
            item['latency'] = latency
        
        except Exception as e:
            print(f"ReAct Error for ID {item.get('id')}: {e}")
            item['prediction'] = f"Error: {str(e)}"
        
        item['tokens'] = total_tokens
        results.append(item)

# 6. 최종 결과를 동적 파일 이름으로 저장
output_filename = f't1_{args.method}_results.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n작업 완료. 결과가 '{output_filename}' 파일에 저장되었습니다.")