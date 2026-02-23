import json
import csv
import time
from typing import List, Dict
from ..utils.agent import ABSASystem 
from tqdm import tqdm
import os

def load_test_cases(filename: str) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results_to_csv(results: List[Dict], filename: str):
    fieldnames = ['id', 'input_text', 'difficulty', 'status', 'attempts', 'final_extraction', 'notes']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            # Flatten the extraction list to a string for CSV storage
            res['final_extraction'] = json.dumps(res['final_extraction'], ensure_ascii=False)
            writer.writerow(res)
    print(f"\nResults saved to {filename}")

def main():
    model_path = "Qwen/Qwen3-4B" 
    print(f"Initializing ABSA System with {model_path}...")
    system = ABSASystem(model_path, prompts_dir="prompts")

    # Load data
    test_case_path = 'dataset/hoasa_hotel/indo/mvp_aos/test.json'
    with open(test_case_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    # Take first 5 test cases for quick testing
    test_cases = test_cases[4:10]
    # Remove '[A] [O] [S]' from input text
    for case in test_cases:
        case['input'] = case['input'].replace('[A] [O] [S]', '').strip()
    results = []

    print(f"\nStarting benchmark on {len(test_cases)} test cases...\n")

    # Run extractor-evaluator loop for each test case
    for case in tqdm(test_cases, desc="Processing test cases"):
        print(f"--- Case ID {case['sentence_id']} ---")
        start_time = time.time()
        
        # Run agent loop
        output = system.process_review(case['input'])
        
        elapsed = time.time() - start_time
        print(f"    Finished in {elapsed:.2f}s")

        results.append({
            'id': case['sentence_id'],
            'input_text': case['input'],
            'status': output['status'],
            'attempts': output['attempts'],
            'final_extraction': output['final_output'],
            'notes': case.get('notes', '')
        })

    # Save to dir
    os.makedirs('results', exist_ok=True)
    with open('results/absa_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()