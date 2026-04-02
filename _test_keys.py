from google import genai
import config
import time

models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]

for i, key in enumerate(config.GEMINI_API_KEYS):
    print(f"\n--- Key {i+1} (starts: {key[:12]}...) ---")
    client = genai.Client(api_key=key)
    for model in models:
        try:
            r = client.models.generate_content(
                model=model,
                contents="Say OK"
            )
            print(f"  {model}: OK — {r.text.strip()[:30]}")
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"  {model}: QUOTA EXHAUSTED")
            elif "404" in err:
                print(f"  {model}: MODEL NOT FOUND")
            elif "403" in err:
                print(f"  {model}: API KEY INVALID/DISABLED")
            else:
                print(f"  {model}: ERROR — {err[:80]}")
        time.sleep(2)
