"""Token counts and code-similarity metrics."""
import difflib
import tokenize  # only defined for Python code
from io import BytesIO

import tiktoken


def token_count(doc_text):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    encoding = tiktoken.Encoding(
        name="gpt-35-turbo",
        pat_str=tokenizer._pat_str,
        mergeable_ranks=tokenizer._mergeable_ranks,
        special_tokens={
            **tokenizer._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
        },
    )

    tokens_doc = encoding.encode(
        doc_text, allowed_special={"<|im_start|>", "<|im_end|>"}
    )

    return len(tokens_doc)
    
def tokenize_by_tiktoken(code):
    # Get an encoding (e.g., "cl100k_base")
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Encode and decode a sample text
    encoded_tokens = enc.encode(code)
    decoded_code = enc.decode(encoded_tokens)

    # Verify that the decoding works correctly
    assert decoded_code == code

    return encoded_tokens

def tokenize_by_tokenize(code):
    # Includes an error handling to "skip" over bad token conversions
    tokens = []
    try:
        readline = BytesIO(code.encode('utf-8')).readline
        token_gen = tokenize.tokenize(readline)
        while True:
            try:
                token = next(token_gen)
                tokens.append(token)
            except IndentationError as e:
                print(f"Indentation error in the code: {e}")
                continue
            except StopIteration:
                break
    except Exception as e:
        print(f"Unexpected error: {e}")

    return tokens

def tokenize_code(code, method="tokenize"):

    if method == "tokenize":
        tokens = tokenize_by_tokenize(code)
    elif method == "tiktoken":
        tokens = tokenize_by_tiktoken(code)
    
    return tokens
    
def token_similarity(code1, code2, method):
    tokens1 = tokenize_code(code1, method)
    tokens2 = tokenize_code(code2, method)
    sequence_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    return sequence_matcher.ratio()

def find_line_in_lines(line, lines):
    """Helper function to find a line in a list of lines."""
    for i, l in enumerate(lines):
        if line.strip() == l.strip():
            return i
    return -1

def compare_lines_with_reordering(code1, code2):
    """Compare lines of code1 with lines of code2 allowing reordering."""
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    
    matched_lines = 0
    for line1 in lines1:
        index = find_line_in_lines(line1, lines2)
        if index != -1:
            matched_lines += 1
            lines2.pop(index)  # Remove the matched line to prevent duplicate matching
    
    total_lines = max(len(lines1), len(code2.splitlines()))
    similarity_ratio = matched_lines / total_lines if total_lines > 0 else 0
    return similarity_ratio
