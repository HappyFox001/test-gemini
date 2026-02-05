from google import genai
import time

client = genai.Client()

start_time = time.time()
print(f"Starting content generation at t=0.000s\n")

response = client.models.generate_content_stream(
    model="gemini-3-flash-preview",
    contents=["Explain how AI works"]
)

chunk_count = 0
for chunk in response:
    chunk_count += 1
    elapsed_time = time.time() - start_time
    print()
    print(f"[Chunk {chunk_count} at t={elapsed_time:.3f}s]: ", end="")
    print(chunk.text, end="")

total_time = time.time() - start_time
print(f"\n\nTotal time: {total_time:.3f}s")
print(f"Total chunks: {chunk_count}")