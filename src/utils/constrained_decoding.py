from transformers import LogitsProcessor, AutoTokenizer
import torch

# Re-implemented and modified from [MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction](https://aclanthology.org/2023.acl-long.240/)

class BaseConstrainedDecoder(LogitsProcessor):
	'''
	A custom logits processor that modifies the logits during generation.
	All tokens that are not in the input sequence will be masked out (set to -inf) to prevent the model from generating them.
	Exception for special tokens like eos_token, etc.
	Can handle both batched and unbatched input.
	'''
	def __init__(self, input_ids: torch.LongTensor, tokenizer: AutoTokenizer):
		super().__init__()
		self.input_ids = input_ids
		self.tokenizer = tokenizer
		self.special_token_ids = set(tokenizer.all_special_ids)
		self.special_words = set(tokenizer(['positive', 'negative', ' positive', 'Ġpositive', ' negative', 'Ġnegative', 'null', ' null', 'Ġnull'], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)['input_ids'].reshape(-1).tolist())
		self.spaced_first_word = None
	
	def _prepare_batch_allowed_tokens(self, tokenizer: AutoTokenizer) -> None:
		
		# Pre-compute allowed token ids for each batch item
		if self.input_ids.dim() == 1:
			# Unbatched: add batch dimension
			self.input_ids = self.input_ids.unsqueeze(0)
		
		# Create a set of allowed tokens for each batch item
		self.batch_allowed_tokens = []
		for i in range(self.input_ids.size(0)):

			# Handle spaced first word
			list_of_tokens = self.input_ids[i].tolist()
			words = tokenizer.decode(list_of_tokens, skip_special_tokens=True)
			words_split = words.split(' ')
			self.spaced_first_word = f' {words_split[0]}'

			# Generate all substrings of the first word to handle some typos in tokenization
			substrings = set()
			for i in range(len(words_split[0])):
				for j in range(i + 1, len(words_split[0]) + 1):
					substrings.add(words_split[0][i:j])
					if i == 0:
						substrings.add(' ' + words_split[0][i:j])
			
			# Tokenizer all substrings, and filter them based on this rule: only keep those that are tokenized as a single token
			tokenized_substrings = set()
			for substring in substrings:
				tokenized = tokenizer(substring, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
				if tokenized['input_ids'].size(1) == 1:
					tokenized_substrings.add(tokenized['input_ids'].item())
				
			# spaced_token_id = set(tokenizer([self.spaced_first_word, f'Ġ{words_split[0]}'], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)['input_ids'].reshape(-1).tolist())

			# Combine all allowed tokens
			allowed_tokens = set(list_of_tokens).union(self.special_token_ids).union(self.special_words).union(tokenized_substrings)

			# Store the allowed tokens for this batch item
			self.batch_allowed_tokens.append(allowed_tokens)
	
	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		batch_size = scores.size(0)
		
		# Create a mask initialized to -inf for all tokens
		mask = torch.full_like(scores, float('-inf'))
		
		# Apply mask for each batch item
		for i in range(batch_size):
			# Handle case where batch size might be smaller than pre-computed
			allowed_idx = min(i, len(self.batch_allowed_tokens) - 1)
			allowed_token_ids = self.batch_allowed_tokens[allowed_idx]
			
			# Convert allowed token IDs to a list and set their mask values to 0
			allowed_ids_list = list(allowed_token_ids)
			mask[i, allowed_ids_list] = 0.0
		
		# Apply the mask to the scores
		modified_scores = scores + mask
		return modified_scores

class MVPConstrainedDecoder(BaseConstrainedDecoder):
	'''
	A custom logits processor for the MVP dataset that modifies the logits during generation.
	All tokens that are not in the input sequence will be masked out (set to -inf) to prevent the model from generating them.
	Exception for special tokens like eos_token, etc.
	Can handle both batched and unbatched input.
	'''
	def __init__(self, input_ids: torch.LongTensor, tokenizer: AutoTokenizer):
		super().__init__(input_ids, tokenizer)
		self.special_words = self.special_words.union(set(tokenizer([' [SSEP] '], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)['input_ids'].reshape(-1).tolist()))
		self._prepare_batch_allowed_tokens(tokenizer)

class GASConstrainedDecoder(BaseConstrainedDecoder):
	'''
	A custom logits processor for the GAS dataset that modifies the logits during generation.
	All tokens that are not in the input sequence will be masked out (set to -inf) to prevent the model from generating them.
	Exception for special tokens like eos_token, etc.
	Can handle both batched and unbatched input.
	'''
	def __init__(self, input_ids: torch.LongTensor, tokenizer: AutoTokenizer):
		super().__init__(input_ids, tokenizer)
		temp_special_words = ['|' , ' |', '| ', ' | ', ';' , ' ;', '; ', ' ; ', '(', ' (', '( ', ' ( ', ')', ' )', ') ', ' ) ']
		self.special_words = self.special_words.union(set(tokenizer(temp_special_words, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)['input_ids'].reshape(-1).tolist()))
		self._prepare_batch_allowed_tokens(tokenizer)

class LegoABSAConstrainedDecoder(BaseConstrainedDecoder):
	'''
	A custom logits processor for the LegoABSA dataset that modifies the logits during generation.
	All tokens that are not in the input sequence will be masked out (set to -inf) to prevent the model from generating them.
	Exception for special tokens like eos_token, etc.
	Can handle both batched and unbatched input.
	'''
	def __init__(self, input_ids: torch.LongTensor, tokenizer: AutoTokenizer):
		super().__init__(input_ids, tokenizer)
		self.special_words = self.special_words.union(set(tokenizer(['<|aspect|>', '<|opinion|>', '<|sentiment|>', ' ', ';'], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)['input_ids'].reshape(-1).tolist()))
		self._prepare_batch_allowed_tokens(tokenizer)