from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Split user input into multiple atomic predicates
def split_joined_predicates(prompt: str) -> list:
    system_prompt = (
        "You are an assistant that splits a visual search request into multiple standalone visual predicates. "
        "Each predicate should describe a single visual item or scene clearly. "
        "Do not return a sentence. Just return a Python list of short phrases."
    )

    user_prompt = f"""
User input: "{prompt}"

Output a Python list of concise, separate visual predicates. For example:
→ ["apples in the market", "peanuts at some stand"]

Only return the Python list. No explanation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        result = eval(content)
        return result

    except Exception as e:
        print("❌ LLM error while splitting predicates:", e)
        return []