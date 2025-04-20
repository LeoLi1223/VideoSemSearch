from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def split_joined_predicates(prompt: str) -> tuple[list[str], list[str]]:
    system_prompt = """
        You are a helpful assistant for decomposing natural language video search prompts.

        Given a natural language query that may describe:
        - what the user wants to see (positive visual scenes), and
        - what they want to avoid (negative visual elements),

        return a JSON object in the form:

        {
        "include": [...],  // short descriptions of what to include
        "exclude": [...]   // short descriptions of what to avoid
        }

        Each entry should be a concise visual predicate that can be understood by a vision-language model.

        ✅ If a sentence contains multiple actions or attributes tied to the same subject (e.g. "Asian man"), you must repeat the subject in each entry:
        - "Asian man wearing a cap without glasses walking through an airport"
        → include: ["Asian man wearing a cap", "Asian man walking through an airport"]
        → exclude: ["Asian man with glasses"]

        🔍 Also infer negatives from implicit phrases like:
        - "without glasses" → exclude: "glasses"
        - "not holding a phone" → exclude: "phone"

        📌 Examples:

        Input:
        "I want apples or peanuts, but not lemons"
        Output:
        {"include": ["apples", "peanuts"], "exclude": ["lemons"]}

        Input:
        "Asian man wearing a cap without glasses walks through an airport"
        Output:
        {"include": ["Asian man wearing a cap", "Asian man walking through an airport"], "exclude": ["Asian man with glasses"]}

        Return only the JSON object. Do not include explanation or natural language commentary.
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
    