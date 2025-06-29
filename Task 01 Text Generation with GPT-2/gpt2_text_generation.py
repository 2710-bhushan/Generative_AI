from transformers import pipeline

text_generator = pipeline('text-generation', model='./gpt2-finetuned')

output = text_generator("Once upon a time", max_length=100, num_return_sequences=1)
print(output[0]['generated_text'])
