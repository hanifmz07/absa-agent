import json
import re
import os
from typing import List, Dict, Any
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class ABSASystem:
	def __init__(self, model_name: str, prompts_dir: str = "prompts", max_model_len: int = 4096):
		# Load prompts
		print(f"Loading prompts from directory: {prompts_dir}/")
		self.prompts = self._load_prompts_from_dir(prompts_dir)
		
		# Load LLM and tokenizer
		print(f"Loading vLLM model: {model_name}")
		self.llm = LLM(
			model=model_name, 
			trust_remote_code=True,
			# max_model_len=max_model_len,
			dtype=torch.bfloat16,
			gpu_memory_utilization=0.9
		)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		
		# Sampling params
		self.extractor_params = SamplingParams(temperature=0.6, max_tokens=2048, top_p=0.95)
		self.evaluator_params = SamplingParams(temperature=0.6, max_tokens=2048, top_p=0.95)

	def _load_prompts_from_dir(self, directory: str) -> Dict:
		"""Loads .md files into a dictionary."""
		prompts = {}
		required_files = {
			"extractor_system": "extractor_system.md",
			"extractor_user": "extractor_user.md",
			"evaluator_system": "evaluator_system.md",
			"evaluator_user": "evaluator_user.md",
			"critique_template": "critique_template.md"
		}

		for key, filename in required_files.items():
			path = os.path.join(directory, filename)
			if not os.path.exists(path):
				raise FileNotFoundError(f"Missing required prompt file: {path}")
			
			with open(path, 'r', encoding='utf-8') as f:
				prompts[key] = f.read().strip()
		
		return prompts

	def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt}
		]
		return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

	def _parse_json(self, text: str) -> Any:
		try:
			clean_text = re.sub(r"```json|```", "", text).strip()
			return json.loads(clean_text)
		except json.JSONDecodeError:
			return None

	def _parse_reasoning_output(self,output_text):
		"""
		Parses Qwen3 output to separate the 'Thinking' process from the 'Final Answer'.
		Returns a dictionary with 'reasoning' and 'content'.
		"""
		# Regex to capture content inside <think> tags
		# Qwen3 typically uses <think> ... </think>
		reasoning_pattern = r"<think>(.*?)</think>"
		
		match = re.search(reasoning_pattern, output_text, re.DOTALL)
		
		if match:
			reasoning = match.group(1).strip()
			# The content is everything after the closing </think> tag
			content = output_text.split("</think>")[-1].strip()
		else:
			# Fallback if no thinking tags are found (e.g., model decided not to think)
			reasoning = None
			content = output_text.strip()
			
		return {"reasoning": reasoning, "content": content}

	def run_extractor(self, input_text: str, critique: str = "") -> List[Dict]:
		# Prepare eval/critique
		critique_text = ""
		if critique:
			critique_text = self.prompts["critique_template"].format(critique=critique)

		# Replace variables in the user prompt
		user_prompt_filled = self.prompts["extractor_user"].format(
			input_text=input_text, 
			critique_instruction=critique_text
		)

		full_prompt = self._format_prompt(self.prompts["extractor_system"], user_prompt_filled)
		
		# Generate
		outputs = self.llm.generate([full_prompt], self.extractor_params, use_tqdm=False)
		result = self._parse_json(outputs[0].outputs[0].text)
		
		if not isinstance(result, list):
			return [{"error": "Invalid JSON format"}]
		return result

	def run_evaluator(self, input_text: str, extraction: List[Dict]) -> Dict:
		# Replace variables in the user prompt
		user_prompt_filled = self.prompts["evaluator_user"].format(
			input_text=input_text,
			extracted_json=json.dumps(extraction, indent=2)
		)

		full_prompt = self._format_prompt(self.prompts["evaluator_system"], user_prompt_filled)
		
		# Generate
		outputs = self.llm.generate([full_prompt], self.evaluator_params, use_tqdm=False)
		result = self._parse_json(outputs[0].outputs[0].text)
		
		if not isinstance(result, dict) or "is_correct" not in result:
			return {"is_correct": True, "reasoning": "Parser failed", "critique": ""}
		return result

	def process_review(self, input_text: str, max_retries: int = 3) -> Dict:
		current_critique = ""
		history = []
		
		print(f"\nProcessing: length={len(input_text)} chars")
		
		for attempt in range(max_retries):
			extraction = self.run_extractor(input_text, current_critique)
			evaluation = self.run_evaluator(input_text, extraction)
			
			history.append({
				"attempt": attempt + 1,
				"extraction": extraction,
				"evaluation": evaluation
			})

			if evaluation.get("is_correct") is True:
				print(f"  [✓] Attempt {attempt+1} Successful.")
				return {
					"final_output": extraction,
					"status": "success",
					"attempts": attempt + 1,
					"history": history
				}
			
			current_critique = evaluation.get("critique", "Incorrect extraction.")
			print(f"  [x] Attempt {attempt+1} Rejected. Critique: {current_critique}")

		print("  [!] Max retries reached.")
		return {
			"final_output": extraction,
			"status": "failed",
			"attempts": max_retries,
			"history": history
		}

if __name__ == "__main__":
	pass