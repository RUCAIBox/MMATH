import os
import json
import pandas as pd
import re
from tqdm import tqdm
import fasttext

# Load fastText language ID model
model = fasttext.load_model("lid.176.bin")

# Detect language function
def detect_language(text):
    predictions = model.predict(text, k=1)
    lang_code = predictions[0][0].replace('__label__', '')
    return lang_code

def process_prediction(prediction):
    return prediction.replace('\n', ' ')

model_list_full = [
    "Qwen/Qwen2.5-7B-Instruct",
    # ...
]

LANGUAGE = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']
n = 4  # number of trials per question

results = []

for model_path in model_list_full:
    if 'checkpoint' in model_path:
        model_name = model_path.split("/")[-2] + '/' + model_path.split("/")[-1]
    else:
        model_name = model_path.split("/")[-1]
        
    result_dir = os.path.join("results", model_name)

    # First: Augment results with detected language info
    for lang in LANGUAGE:
        res_file = os.path.join(result_dir, f"{lang}.json")
        if not os.path.exists(res_file):
            print(f"{res_file} not exists.")
            continue
        
        res = json.load(open(res_file))
        modified = False
        
        for r in tqdm(res, desc=f"DetectLang-{model_name}-{lang}"):
            for j in range(n):
                prediction_j = r.get(f'prediction_{j}')
                if prediction_j is None:
                    raise ValueError(f"prediction_{j} is None.")

                if '</think>' in prediction_j:
                    think_content = ' '.join(prediction_j.split('</think>')[0:-1])
                    ans_content = prediction_j.split('</think>')[-1]
                else:
                    think_content = prediction_j
                    ans_content = prediction_j

                r[f'think_content_{j}'] = think_content.strip()
                r[f'ans_content_{j}'] = ans_content.strip()
                r[f'detect_think_lang_{j}'] = detect_language(process_prediction(think_content))
                r[f'detect_ans_lang_{j}'] = detect_language(process_prediction(ans_content))
                r[f'detect_lang_{j}'] = detect_language(process_prediction(prediction_j))
                modified = True
        
        if modified:
            with open(res_file, 'w') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)

    # Then: Compute consistency
    total_think = 0
    correct_think = 0
    total_ans = 0
    correct_ans = 0
    
    for lang in LANGUAGE:
        res_file = os.path.join(result_dir, f"{lang}.json")
        if not os.path.exists(res_file):
            continue
        
        res = json.load(open(res_file))
        
        for r in tqdm(res, desc=f"LCR-{model_name}-{lang}"):
            for j in range(n):
                think_lang = r.get(f'detect_think_lang_{j}', '')
                ans_lang = r.get(f'detect_ans_lang_{j}', '')
                
                if think_lang != '':
                    total_think += 1
                    if think_lang == lang:
                        correct_think += 1
                if ans_lang != '':
                    total_ans += 1
                    if ans_lang == lang:
                        correct_ans += 1
    
    think_consistency = (correct_think / total_think) * 100 if total_think > 0 else 0
    ans_consistency = (correct_ans / total_ans) * 100 if total_ans > 0 else 0

    results.append({
        "Model": model_name.replace("/", "-"),
        "Thinking LCR (%)": round(think_consistency, 2),
        "Answering LCR (%)": round(ans_consistency, 2)
    })

# Output LaTeX Table
results_df = pd.DataFrame(results)
latex_table = []
latex_table.append("\\begin{table}[htbp]")
latex_table.append("\\centering")
latex_table.append("\\scalebox{0.65}{")
latex_table.append("\\begin{tabular}{lcc}")
latex_table.append("\\toprule")
latex_table.append("Model & Thinking LCR (\\%) & Answering LCR (\\%) \\\\")
latex_table.append("\\midrule")

for _, row in results_df.iterrows():
    model = row["Model"]
    think_lcr = row["Thinking LCR (%)"]
    ans_lcr = row["Answering LCR (%)"]
    latex_table.append(f"{model} & {think_lcr:.2f} & {ans_lcr:.2f} \\\\")

latex_table.append("\\bottomrule")
latex_table.append("\\end{tabular}}")
latex_table.append("\\caption{Language Consistency Rate (LCR) for different models. Thinking LCR measures the match between detected thinking language and question language; Answering LCR measures the match for the answer language.}")
latex_table.append("\\label{tab:lcr_results}")
latex_table.append("\\end{table}")

print("\n".join(latex_table))
