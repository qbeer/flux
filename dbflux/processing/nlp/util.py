
from typing import List, Tuple


def get_token_span_from_char_span(input_string: str, input_tokens: List[str], span_start: int, span_end: int) -> Tuple[int, int]:
    # Word span start/end - is a bit tricky to parse, but luckily we should only have
    # to do this once
    word_span_start = -1
    word_span_end = -1

    # This is a pretty little piece of code which 
    # tries to match tokens to character indices by scanning 
    # across the tokens. 
    current_token = 0
    current_token_idx = 0
    incrementing = False
    for idx, char in enumerate(input_string):
        # If we're not incrementing, and we match the first character of the current token
        # we need to start incrementing
        if not incrementing and input_tokens[current_token][current_token_idx] == char:
            incrementing = True

        # If our index is equal to one of the token span starts, we need to click up                            
        if idx == span_start:
            word_span_start = current_token
        if idx == span_end:
            word_span_end = current_token + 1
            break
        
        # If we're incrementing, we should match the tokens!
        if incrementing:
            current_token_idx += 1

        # If we have looped through an entire token, turn off incrementing, and reset the token
        # indices
        if current_token_idx >= len(input_tokens[current_token]):
            incrementing = False
            current_token_idx = 0
            current_token += 1

    return (word_span_start, word_span_end)