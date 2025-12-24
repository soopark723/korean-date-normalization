# %%
# pip3 install openai python-dotenv python-dateutil

# %%
import os
import json
import re
import time
from pathlib import Path
from openai import OpenAI
from datetime import datetime, date
from dotenv import load_dotenv
from collections import Counter
import dateutil.parser

# %%
# -------------------- LOAD API KEY -----------------------
# Load environment variables from .env file
load_dotenv()  

# Initialize client once (outside the function)
client = OpenAI(api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1")

# %%
# -------------------- HELPERS ----------------------------
ISO_DAY = "%Y-%m-%d"

def parse_any_date(date_input):
    """Parse many common date formats into YYYY-MM-DD string.
    Returns None if parsing fails.
    Accepts datetime/date/strings.
    """
    if date_input is None:
        return None
    if isinstance(date_input, (datetime, date)):
        return date_input.date().strftime(ISO_DAY)
    s = str(date_input).strip()
    if not s:
        return None
    # Try ISO first
    try:
        dt = dateutil.parser.isoparse(s)
        return dt.date().strftime(ISO_DAY)
    except Exception:
        pass
    # Fallback to generic parse
    try:
        dt = dateutil.parser.parse(s, dayfirst=False, yearfirst=False)
        return dt.date().strftime(ISO_DAY)
    except Exception:
        return None


def safe_extract_json(text: str) -> str:
    """Return the first balanced JSON object or array substring found in text.
    If none found, return the original text (so downstream heuristics can try).
    """
    if not isinstance(text, str):
        return text
    # Look for first balanced { ... } or [ ... ]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    # Regex fallback: first {...} or [...] chunk
    m = re.search(r'(\{.*?\}|\[.*?\])', text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text


def normalize_prediction(pred):
    """Always return a list of strings (raw tokens). Conservative: flatten lists or JSON.
    """
    if pred is None:
        return []
    # If it's already a list, flatten and stringize
    if isinstance(pred, list):
        out = []
        for x in pred:
            if x is None:
                continue
            if isinstance(x, (str, int, float)):
                out.append(str(x).strip())
            else:
                try:
                    out.append(json.dumps(x, ensure_ascii=False))
                except Exception:
                    out.append(str(x))
        return [s for s in out if s]
    # If it's a dict, try to extract known keys
    if isinstance(pred, dict):
        # Common: {"prediction": [...]} or similar
        if "prediction" in pred:
            return normalize_prediction(pred["prediction"])
        # flatten dict to single string
        try:
            return [json.dumps(pred, ensure_ascii=False)]
        except Exception:
            return [str(pred)]
    # If it's a simple scalar, try to interpret string
    s = str(pred).strip()
    # If it looks like JSON, try to load it
    if s.startswith("{") or s.startswith("["):
        try:
            j = json.loads(s)
            return normalize_prediction(j)
        except Exception:
            pass
    # Split by common separators (newline, comma, semicolon)
    tokens = re.split(r'[\n,;]+', s)
    return [t.strip() for t in tokens if t.strip()]


def validate_dates(pred_list):
    """Normalize tokens to ISO date strings. Return (valid_dates, invalid_tokens).
    """
    parsed = []
    invalid = []
    for tok in pred_list:
        iso = parse_any_date(tok)
        if iso:
            parsed.append(iso)
        else:
            invalid.append(tok)
    return parsed, invalid

# %%
# ---------------- DATASET LOADING -------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_datasets():
    t1 = load_json("data/t1_dataset.json")
    t2 = load_json("data/t2_dataset.json")
    t3 = load_json("data/t3_dataset.json")
    return t1, t2, t3

# %%
# ---------- SYSTEM MESSAGE ----------
SYSTEM_MESSAGE = """
You are a Korean date and schedule assistant.
You must compute dates using correct Korean calendar rules (Asia/Seoul).
Output ONLY a JSON object with a single key "prediction."
Never output explanations.
Week starts on Monday.
Always compute weekdays from the anchor date, never guess.
"""

# %%
# ---------- COT PROMPT TEMPLATE ----------
cot_prompt_template = """
You are a Korean date and schedule assistant. 
Given the input JSON object containing a natural language query and an anchor date, 
reason step by step to compute the correct date(s) that satisfy all constraints.

- Dates must match the real Korean calendar (Asia/Seoul).
- Never guess weekdays—always compute using the anchor date.
- Korean week starts Monday.
- Interpret “이번 주 / 다음 주 / 지난 주” using Monday as start of week.

[FEW-SHOT EXAMPLES]
{few_shot_examples}

[TEST QUERY]
Input:
{{
  "input_text": "{input_text}",
  "anchor_date": "{anchor_date}"
}}
Thought:
"""

# %%
# ---------------- CoT EVALUATION ------------------------
def evaluate_cot(dataset, few_shot_examples):
    results = []

    for ex in dataset:
        row = {
            'id': ex.get('id'),
            'input_text': ex.get('input_text'),
            'anchor_date': ex.get('anchor_date'),
            'gold_standard': ex.get('gold_standard'),
            'predicted_raw': None,
            'predicted_parsed': [],
            'invalid_predicted': [],
            'correct': False,
            'parse_error': None,
            'normalization_error': None,
            'latency_seconds': None,
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None
        }

        try:
            prompt = cot_prompt_template.format(
                few_shot_examples=few_shot_examples,
                input_text=ex['input_text'].replace('"', '\\"'),
                anchor_date=ex['anchor_date']
            )

            start = time.time()
            response = client.chat.completions.create(
                model='solar-pro2',
                temperature=0,
                stream=False,
                messages=[
                    {'role': 'system', 'content': SYSTEM_MESSAGE},
                    {'role': 'user', 'content': prompt}
                ]
            )
            end = time.time()
            row['latency_seconds'] = end - start

            # Safe usage access
            usage = getattr(response, 'usage', None)
            if usage:
                row['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
                row['completion_tokens'] = getattr(usage, 'completion_tokens', None)
                row['total_tokens'] = getattr(usage, 'total_tokens', None)

            # raw output
            try:
                raw_output = response.choices[0].message.content.strip()
            except Exception:
                raw_output = str(response)
            row['predicted_raw'] = raw_output

            # Extract CoT reasoning
            json_part = safe_extract_json(raw_output)

            cot_reasoning = raw_output.replace(json_part, "").strip()
            # Remove trailing quotes, fences, artifacts if any
            cot_reasoning = cot_reasoning.replace("```json", "").replace("```", "").strip()

            row["cot_reasoning"] = cot_reasoning

            parsed_json = None
            parse_err = None
            try:
                parsed_json = json.loads(json_part)
            except Exception as e:
                # Record parse error but continue to try to normalize from raw
                parse_err = f'json.loads failed: {e}'
            row['parse_error'] = parse_err

            # If parsed_json exists and is dict with `prediction`, use it.
            model_pred = None
            if parsed_json is not None:
                if isinstance(parsed_json, dict) and 'prediction' in parsed_json:
                    model_pred = parsed_json['prediction']
                else:
                    model_pred = parsed_json
            else:
                # fallback: try to find a JSON-like substring again or use raw text
                model_pred = raw_output

            # Normalize prediction into token list
            try:
                pred_tokens = normalize_prediction(model_pred)
            except Exception as e:
                pred_tokens = []
                row['normalization_error'] = f'normalize_prediction failed: {e}'

            # Validate/parse tokens to ISO
            valid_dates, invalid_tokens = validate_dates(pred_tokens)
            row['predicted_parsed'] = valid_dates
            row['invalid_predicted'] = invalid_tokens

            # Normalize gold standard
            gold = ex.get('gold_standard')
            gold_list = gold if isinstance(gold, list) else [gold]
            gold_normalized = []
            gold_errors = []
            for g in gold_list:
                iso = parse_any_date(g)
                if iso:
                    gold_normalized.append(iso)
                else:
                    gold_errors.append(g)
            if gold_errors:
                row['normalization_error'] = (row.get('normalization_error') or '') + f' gold_unparseable: {gold_errors}'

            # Compare as multisets (duplicates matter, order irrelevant)
            row['gold_standard'] = gold_normalized
            row['correct'] = Counter(valid_dates) == Counter(gold_normalized)

            results.append(row)

        except Exception as e:
            # Outer exception: still append row with error info
            row['parse_error'] = f'outer exception: {e}'
            results.append(row)

    acc = sum(1 for r in results if r['correct']) / len(results) if results else 0.0
    return results, acc

# %%
# -------------------- MAIN EXECUTION --------------------------
if __name__ == '__main__':
    # Paths
    Path('results').mkdir(parents=True, exist_ok=True)

    # Load few-shot examples
    with open('prompts/few_shot_cot.txt', 'r', encoding='utf-8') as f:
        few_shot_examples = f.read()

    # Load datasets
    t1, t2, t3 = load_datasets()

    print('🚀 Running CoT Evaluation...')

    t1_res, t1_acc = evaluate_cot(t1, few_shot_examples)
    t2_res, t2_acc = evaluate_cot(t2, few_shot_examples)
    t3_res, t3_acc = evaluate_cot(t3, few_shot_examples)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Save raw results
    with open(f'results/{timestamp}_cot_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            't1': t1_res,
            't2': t2_res,
            't3': t3_res,
            'accuracy': {'t1': t1_acc, 't2': t2_acc, 't3': t3_acc}
        }, f, indent=2, ensure_ascii=False)

    # 2. Latency + tokens summary
    def avg_metric(group, metric):
        values = [r.get(metric) for r in group if r.get(metric) is not None]
        return sum(values) / len(values) if values else None

    summary = {
        'accuracy': {'t1': t1_acc, 't2': t2_acc, 't3': t3_acc},
        'latency_seconds': {
            't1': avg_metric(t1_res, 'latency_seconds'),
            't2': avg_metric(t2_res, 'latency_seconds'),
            't3': avg_metric(t3_res, 'latency_seconds')
        },
        'tokens': {
            't1': {
                'prompt_avg': avg_metric(t1_res, 'prompt_tokens'),
                'completion_avg': avg_metric(t1_res, 'completion_tokens'),
                'total_avg': avg_metric(t1_res, 'total_tokens')
            },
            't2': {
                'prompt_avg': avg_metric(t2_res, 'prompt_tokens'),
                'completion_avg': avg_metric(t2_res, 'completion_tokens'),
                'total_avg': avg_metric(t2_res, 'total_tokens')
            },
            't3': {
                'prompt_avg': avg_metric(t3_res, 'prompt_tokens'),
                'completion_avg': avg_metric(t3_res, 'completion_tokens'),
                'total_avg': avg_metric(t3_res, 'total_tokens')
            }
        }
    }

    with open(f'results/{timestamp}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3. Save raw reasoning text (predicted_raw grouped by ID)
    with open(f'results/{timestamp}_raw_reasoning.txt', 'w', encoding='utf-8') as f:
        for name, group in [('T1', t1_res), ('T2', t2_res), ('T3', t3_res)]:
            f.write(f"\n===== {name} =====\n\n")
            for r in group:
                f.write(f"ID: {r.get('id')}\n")

                f.write("RAW OUTPUT:\n")
                f.write((r.get('predicted_raw') or '') + '\n\n')

                f.write("COT REASONING:\n")
                f.write((r.get('cot_reasoning') or '') + '\n')
                
                f.write('-' * 50 + '\n\n')

    # 4. Mismatches only
    mismatches = []
    for name, group in [('T1', t1_res), ('T2', t2_res), ('T3', t3_res)]:
        for r in group:
            if not r.get('correct'):
                mismatches.append({
                    'group': name,
                    'id': r.get('id'),
                    'gold': r.get('gold_standard'),
                    'predicted': r.get('predicted_parsed'),
                    'raw_output': r.get('predicted_raw'),
                    'cot_reasoning': r.get('cot_reasoning'),
                    'tokens': {
                        'prompt': r.get('prompt_tokens'),
                        'completion': r.get('completion_tokens'),
                        'total': r.get('total_tokens')
                    },
                    'parse_error': r.get('parse_error'),
                    'normalization_error': r.get('normalization_error')
                })

    with open(f'results/{timestamp}_mismatches.json', 'w', encoding='utf-8') as f:
        json.dump(mismatches, f, indent=2, ensure_ascii=False)

    print('✨ Evaluation complete. All files saved in /results/')


