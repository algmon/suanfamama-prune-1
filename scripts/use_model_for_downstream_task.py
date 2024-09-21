# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="baffo32/decapoda-research-llama-7B-hf",
    device_map='auto',
    torch_dtype=torch.float16
)

# Generate text
generated_text = pipe("Once upon a time in a distant land,", max_length=100, num_return_sequences=1)

print("Generated Text:")
print(generated_text[0]['generated_text'])