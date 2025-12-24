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

def execute_tool_with_llm(client: OpenAI, tool_name: str, tool_input: any) -> str:
    """
    LLM을 사용하여 주어진 도구의 실행을 시뮬레이션하고 결과를 반환합니다.
    """

    tool_execution_system_prompt = """
You are an expert tool executor. Your task is to act as a specific tool and provide only the direct output for the given input. Do not provide any explanations, apologies, or extra text. Just return the result.

- If the tool is 'calculator', perform the date calculation and return the date string 'YYYY-MM-DD'.
- If the tool is 'calendar_db', act like a database query. Return a JSON array of holidays for the given year/month, or return the exact string "No special days found." if there are none.
- If the tool is 'search', act like a search engine and provide a concise, one-sentence factual answer.
"""
    
    user_prompt = f"Tool: [{tool_name}]\nInput: {json.dumps(tool_input, ensure_ascii=False)}"

    try:
        messages = [
            {"role": "system", "content": tool_execution_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=messages,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        return f"LLM-based tool execution error: {str(e)}"


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
    api_key=os.getenv("UPSTAGE_API_KEY", "up_lqeAS9juLDsEpy8rPfYUNhBY36K1O"),
    base_url="https://api.upstage.ai/v1"
)

# --- 2. 메소드에 따라 프롬프트 로드 ---
system_prompt = ""
observation_prompt = ""

if args.method == 'cot':
    prompt_filepath = '/workspace/NLP/prompts/t3_cot.txt'
    try:
        with open(prompt_filepath, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()
elif args.method == 'react':
    thought_prompt_filepath = '/workspace/NLP/prompts/t3_react_thought.txt'
    try:
        with open(thought_prompt_filepath, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{thought_prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()
        
    observation_prompt_filepath = '/workspace/NLP/prompts/t3_react_observation.txt'
    try:
        with open(observation_prompt_filepath, 'r', encoding='utf-8') as f:
            observation_prompt = f.read()
    except FileNotFoundError:
        print(f"오류: '{observation_prompt_filepath}' 파일을 찾을 수 없습니다.")
        exit()

# 3. 데이터셋 불러오기
try:
    with open('/workspace/NLP/data/T3_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except FileNotFoundError:
    print("오류: '/workspace/NLP/data/T3_dataset.json' 파일을 찾을 수 없습니다.")
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
                model="solar-pro2", messages=messages, temperature=0, response_format={"type": "json_object"}
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
        start_time = time.time()
        total_tokens = 0
        tool_log = []
        current_summary_thought = "" 

        try:
            # --- 루프 시작 (최대 10턴) ---
            for turn in range(10):
                # [Thought: Decide Tool]
                thought_input = {
                    "user_query": input_text,
                    "anchor_date": anchor_date,
                    "current_summary_thought": current_summary_thought
                }
                messages_thought = [
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": json.dumps(thought_input, ensure_ascii=False, indent=2)}
                ]
                response_thought = client.chat.completions.create(
                    model="solar-pro2", messages=messages_thought, temperature=0, response_format={"type": "json_object"}
                )
                total_tokens += getattr(response_thought.usage, "total_tokens", 0)
                thought_output = json.loads(response_thought.choices[0].message.content)

                tool_name = thought_output.get("tool")
                tool_input = thought_output.get("tool_input")
                
                # [Action: Execute Tool using LLM]
                if tool_name in ["calculator", "calendar_db", "search"]:
                    observation = execute_tool_with_llm(client, tool_name, tool_input)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'"

                current_log_entry = {"thought": thought_output.get("thought"), "tool": tool_name, "input": tool_input, "observation": observation}
                tool_log.append(current_log_entry)
                item[f'react_turn_{turn+1}'] = current_log_entry

                # [Observation: Evaluate State & Decide Termination]
                observation_input = {"input_text": input_text, "tool_log": tool_log}
                messages_obs = [
                    {"role": "system", "content": observation_prompt}, 
                    {"role": "user", "content": json.dumps(observation_input, ensure_ascii=False, indent=2)}
                ]
                response_obs = client.chat.completions.create(
                    model="solar-pro2", messages=messages_obs, temperature=0, response_format={"type": "json_object"}
                )
                total_tokens += getattr(response_obs.usage, "total_tokens", 0)
                obs_output = json.loads(response_obs.choices[0].message.content)

                # --- 새로운 출력 형식 처리 로직 ---
                status_array = obs_output.get("status")
                current_summary_thought = obs_output.get("thought")

                if isinstance(status_array, list) and len(status_array) == 2:
                    status_decision = status_array[0]
                    prediction_list = status_array[1]

                    if status_decision == "finish":
                        item['prediction'] = prediction_list
                        item['thought'] = current_summary_thought
                        break # 루프 종료
                else:
                    # 예상치 못한 형식의 응답이 오면 오류 처리 후 루프 종료
                    item['prediction'] = f"Error: Invalid status format from observation: {status_array}"
                    item['thought'] = current_summary_thought
                    break
            else: # for 루프가 break 없이 정상적으로 종료되었을 때 실행
                item['prediction'] = "Error: Reached max turns (10) without finishing."
                item['thought'] = current_summary_thought
            
            latency = time.time() - start_time
            item['latency'] = latency

        except Exception as e:
            print(f"ReAct Error for ID {item.get('id')}: {e}")
            item['prediction'] = f"Error: {str(e)}"
        
        item['tokens'] = total_tokens
        results.append(item)

# 6. 최종 결과를 동적 파일 이름으로 저장
output_filename = f't3_{args.method}_results_llm_tools.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n작업 완료. 결과가 '{output_filename}' 파일에 저장되었습니다.")