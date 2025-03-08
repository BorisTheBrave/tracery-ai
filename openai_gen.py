# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
# ]
# ///
import openai

# Ensure you have openai>=1.0.0
api_key = open("openai_key.txt", "r").read()
client = openai.OpenAI(api_key=api_key)

# Define your CFG in Rail format
rail_grammar = """
start: expr;

expr: number | ("(" expr op expr ")");

op: "+" | "-" | "*" | "/";

number: /[0-9]+/;
"""

# Call OpenAI API with grammar-based sampling
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "system", "content": "Generate a mathematical expression."}],
    temperature=0.7,
    grammar=rail_grammar  # This is the key!
)

print(response.choices[0].message.content)
