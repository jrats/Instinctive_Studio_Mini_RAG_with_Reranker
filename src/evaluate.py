import requests

API_URL = "http://127.0.0.1:8000/ask"

questions = [
    "What are the main safety objectives of ISO 10218 Parts 1 and 2 in industrial robot systems?", 
    "What is a Performance Level (PL) in functional safety, and why is it important?", 
    "Name two basic methods for machine guarding according to OSHA.", 
    "What is SISTEMA used for in evaluating machinery safety functions?", 
    "How do EN ISO 13849-1 and IEC 62061 differ in their approaches to safety-related control systems?", 
    "Under the EU Machinery Regulation 2023/1230, what role do “safety components” play in achieving conformity assessment?", 
    "Does ISO 10218 include standards for kitchen food safety and hygiene when using robotic arms?", 
    "What are the top 10 most popular robots in science fiction movies?"
]

results = []

for i, q in enumerate(questions, start=1):
    entry = {"Q#": i, "Question": q}

    # Baseline mode
    resp_baseline = requests.post(API_URL, json={"q": q, "k": 3, "mode": "baseline"}).json()
    baseline_answer = resp_baseline.get("answer")
    entry["Baseline_Answer"] = baseline_answer
    entry["Baseline_TopScore"] = resp_baseline.get("top_score")
    entry["Baseline_Contexts"] = [c["title"] for c in resp_baseline.get("contexts", [])]
    entry["Baseline_Abstained"] = baseline_answer is None

    # Hybrid mode
    resp_hybrid = requests.post(API_URL, json={"q": q, "k": 3, "mode": "hybrid"}).json()
    hybrid_answer = resp_hybrid.get("answer")
    entry["Hybrid_Answer"] = hybrid_answer
    entry["Hybrid_TopScore"] = resp_hybrid.get("top_score")
    entry["Hybrid_Contexts"] = [c["title"] for c in resp_hybrid.get("contexts", [])]
    entry["Hybrid_Abstained"] = hybrid_answer is None

    results.append(entry)

# Print nicely
for r in results:
    print("="*80)
    print(f"Q{r['Q#']}: {r['Question']}")
    print(f"Baseline Answer (score {r['Baseline_TopScore']}): {r['Baseline_Answer']}")
    print(f"Baseline Contexts: {r['Baseline_Contexts']}")
    print(f"Baseline Abstained: {r['Baseline_Abstained']}")
    print(f"Hybrid Answer (score {r['Hybrid_TopScore']}): {r['Hybrid_Answer']}")
    print(f"Hybrid Contexts: {r['Hybrid_Contexts']}")
    print(f"Hybrid Abstained: {r['Hybrid_Abstained']}")
