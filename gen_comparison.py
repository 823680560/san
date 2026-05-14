import json
import pandas as pd

# Load dataset
with open("/root/autodl-tmp/san/eval/datasets/law_eval_dataset.json", "r") as f:
    dataset = json.load(f)

# Build qid -> ground_truth_answer lookup
answers_by_qid = {}
for q in dataset["queries"]:
    answers_by_qid[q["qid"]] = {
        "query": q["query"],
        "ground_truth_answer": q["ground_truth_answer"],
    }

# Load generation eval report
with open("/root/autodl-tmp/san/eval/reports/law/generation_eval_results_llama3_1_8b_latest_20260509_1346.json", "r") as f:
    report = json.load(f)

# Build qid -> response lookups for rag and llm_only
rag_by_qid = {r["qid"]: r["response"] for r in report["detailed_results"]["rag"]}
llm_only_by_qid = {r["qid"]: r["response"] for r in report["detailed_results"]["llm_only"]}

# Build result rows
rows = []
for qid in sorted(answers_by_qid.keys()):
    rows.append({
        "编号": qid,
        "query": answers_by_qid[qid]["query"],
        "ground_truth_answer": answers_by_qid[qid]["ground_truth_answer"],
        "response_rag": rag_by_qid.get(qid, ""),
        "response_llm_only": llm_only_by_qid.get(qid, ""),
    })

df = pd.DataFrame(rows)
df.to_excel("/root/autodl-tmp/san/ans_comparation.xlsx", index=False)
print(f"Done. Written {len(rows)} rows to ans_comparation.xlsx")
