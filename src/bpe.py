import pickle
import torch

VOCAB_SIZE = 2000

def bpe(text):
    dictionary = {}
    # Process each character individually, not just words
    for char in text:
        dictionary[char] = dictionary.get(char, 99999)
    
    # Then process the word combinations
    for line in text.splitlines():
        for word in line.split():
            _add_tokens_from_word(dictionary, word)
    
    possible_tokens_list = list(dictionary.items())
    sorted_list = sorted(possible_tokens_list, reverse=True, key=lambda tup: tup[1])
    
    sorted_list_just_tokens = [x[0] for x in sorted_list][0:VOCAB_SIZE-4]

    sorted_list_just_tokens.insert(0, "\n")
    sorted_list_just_tokens.insert(0, " ")
    sorted_list_just_tokens.insert(0, "<<next-token>>")
    sorted_list_just_tokens.insert(0, "<<end-document>>")

    str_to_token = {}
    for i, token in enumerate(sorted_list_just_tokens):
        str_to_token[token] = i

    with open("./token_map", "wb") as f:
        pickle.dump(str_to_token, f)
    with open("./token_list", "wb") as f:
        pickle.dump(sorted_list_just_tokens, f)

def _add_tokens_from_word(dictionary, word):
    possible_token = ""
    for character in word:
        possible_token += character
        if len(possible_token) == 1:
            dictionary[possible_token] = 99999
        elif possible_token in dictionary:
            dictionary[possible_token] += 1
        else:
            dictionary[possible_token] = 1

def load_bpe():
    with open("./token_map", "rb") as f:
        loaded_token_map = pickle.load(f)
    print(loaded_token_map)

    with open("./token_list", "rb") as f:
        loaded_token_list = pickle.load(f)

    print(loaded_token_list)

def tokenize_string(input_string):
    # Load the token map
    with open("./token_map", "rb") as f:
        token_map = pickle.load(f)

    tokens = []
    current_position = 0

    while current_position < len(input_string):
        longest_match = None
        longest_length = 0

        # Try to find the longest matching token starting with maximum possible length
        remaining_text = input_string[current_position:]
        test_length = min(20, len(remaining_text))  # Limit max token length to check
        
        while test_length > 0:
            test_substring = remaining_text[:test_length]
            if test_substring in token_map:
                longest_match = test_substring
                longest_length = test_length
                break
            test_length -= 1

        if longest_match:
            tokens.append(token_map[longest_match])
            current_position += longest_length
        else:
            # If no token matches, move forward by one character
            print("Warning: No token matches for character:", input_string[current_position])
            tokens.append(token_map.get(input_string[current_position], 0))
            current_position += 1

    # Convert to PyTorch tensor
    return torch.tensor(tokens, dtype=torch.long)

if __name__ == "__main__":
    with open("./mingpt/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    bpe(text)
    load_bpe()

    print(tokenize_string("""we have, such. This gallant which thou seest
Was in the wreck; and, but he's something stain'd
With grief that's beauty's canker, thou mightst call him
A goodly person: he hath lost his fellows
And strays about to find 'em.

MIRANDA:
I might call him
A thing divine, for nothing natural
I ever saw so noble.

PROSPERO:

FERDINAND:
Most sure, the goddess
On whom these airs attend! Vouchsafe my prayer
May know if you remain upon this island;
And that you will some good instruction give
How I may bear me here: my prime request,
Which I do last pronounce, is, O you wonder!
If you be maid or no?

MIRANDA:
No wonder, sir;
But certainly a maid.

FERDINAND:
My language! heavens!
I am the best of them that speak this speech,
Were I but where 'tis spoken."""))