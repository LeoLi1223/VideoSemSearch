from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def split_joined_predicates(prompt: str) -> tuple[list[str], list[str]]:
    system_prompt = """
        You are an assistant that extracts exactly what the user *wants* and *doesn't want* to see from a natural language visual query.
        Strictly use the user's original words — do not rephrase, elaborate, or infer additional meanings.

        Return a Python dictionary in the format:
        {
        "include": [...],   # exact phrases the user wants to see
        "exclude": [...]    # exact phrases the user does NOT want to see
        }

        Do not interpret or expand the query (e.g., don't turn "apples" into "apples in a basket").
        Only extract direct phrases from the user input.

        Only return the Python dictionary. No explanation.
        """
    user_prompt = f'User input: "{prompt}"'

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

        # Safe parsing using `ast.literal_eval`
        import ast
        parsed = ast.literal_eval(content)
        return parsed.get("include", []), parsed.get("exclude", [])

    except Exception as e:
        print("Error extracting include/exclude predicates:", e)
        return [], []
    