import random

def build_markov_chain(text, n=1):
    words = text.split()
    index = n
    chain = {}
    for i in range(len(words) - n):
        key = tuple(words[i:i+n])
        next_word = words[i + n]
        chain.setdefault(key, []).append(next_word)
    return chain

def generate_text(chain, length=50):
    key = random.choice(list(chain.keys()))
    result = list(key)
    for _ in range(length):
        next_words = chain.get(key)
        if not next_words:
            break
        next_word = random.choice(next_words)
        result.append(next_word)
        key = tuple(result[-len(key):])
    return ' '.join(result)

# Example Usage
text = "This is a simple text generation example using Markov chains."
chain = build_markov_chain(text, n=1)
print(generate_text(chain))
