import re
from typing import List, Dict, Literal
from .const import ELEMENT2CHAR, CHAR2ELEMENT
def add_space_around_punctuation(text):
    # Except for '-'
    # Ensure space before punctuation
    text = re.sub(r'(\S)([.,!?\(\)\"\';:+/]+)', r'\1 \2', text)
    # Ensure space after punctuation
    text = re.sub(r'([.,!?\(\)\"\';:+/]+)(\S)', r'\1 \2', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Ensure punctuation sequences like '...' are split into spaced dots
    text = re.sub(r'([.]{1,})', lambda m: ' '.join(m.group(1)), text)
    # Ensure punctuation sequences like '!!' or other punctuations that appear consecutively are split into spaced characters
    text = re.sub(r'([!,?\(\)]{1,})', lambda m: ' '.join(m.group(1)), text)
    return text.strip()

def parse_absa_string(text: str):
    """
    Parses a string formatted as "[A] aspect [O] opinion [S] sentiment" into a list of dictionaries.
    Each dictionary contains the tag as the key and the corresponding value.
    For example, "[A] [O] [S] [A] harga [O] terjangkau [S] positive [SSEP] [A] fasilitas [O] nyaman [S] positive" becomes:
    [{'A': 'harga', 'S': 'positive', 'O': 'terjangkau'},
    {'A': 'fasilitas', 'S': 'positive', 'O': 'nyaman'}].

    Args:
        text (str): ABSA string output to be parsed.

    Returns:
        List[Dict[str, str]]: List of dictionaries of parsed ABSA output.

    """
    pattern = r"\[(\w+)\]\s*([^[]+)"
    matches = re.findall(pattern, text)

    result = []
    current_dict = {}

    for tag, content in matches:
        if tag == "SSEP":  # Sentence separator -> Start a new dictionary
            result.append(current_dict)
            current_dict = {}
        else:
            current_dict[CHAR2ELEMENT[tag]] = content.strip()

    if current_dict:  # Append the last sentence if it exists
        result.append(current_dict)

    return result

def convert_to_absa_format(triplets: List[Dict[str, str]], order: List[Literal['A', 'O', 'S']]) -> str:
	"""
	Converts a list of dictionaries containing ABSA triplets into a formatted string.
	Each dictionary should contain keys 'A', 'O', and 'S' for Aspect, Opinion, and Sentiment respectively.
	The order of these elements in the output string is determined by the 'order' parameter.

	Args:
		triplets (List[Dict[str, str]]): List of dictionaries with ABSA triplet information.
	Returns:
		str: A formatted string representing the ABSA triplets.
	"""
	result = []
	for triplet in triplets:
		parts = []
		for key in order:
			if key in triplet:
				parts.append(f"[{ELEMENT2CHAR[key]}] {triplet[key]}")
		result.append(" ".join(parts))
	return " [SSEP] ".join(result)

def lower_triplets(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{k: v.lower() for k, v in triplet.items()} for triplet in triplets]

def punctuation_triplets(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
	return [{k: add_space_around_punctuation(v) for k, v in triplet.items()} for triplet in triplets]