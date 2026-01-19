#Step 1: Define skills (stored outside the prompt)
#These live in files, a DB, or Python dicts — not sent unless selected.

SKILLS = {
    "document_qa": {
        "description": "Answer questions based on provided documents",
        "system_prompt": """
You are a document analysis assistant.
Rules:
- Only answer using the provided document context
- Cite page numbers when possible
- If the answer is not in the documents, say "Not found"
"""
    },

    "contract_analysis": {
        "description": "Analyze legal contracts",
        "system_prompt": """
You are a legal contract analysis assistant.
Rules:
- Identify clauses, risks, obligations
- Be precise and neutral
- Do not provide legal advice
"""
    },

    "summarization": {
        "description": "Summarize long documents",
        "system_prompt": """
You summarize documents concisely.
Rules:
- Focus on key points
- Use bullet points
- No hallucinations
"""
    }
}
#👉 Zero tokens spent so far — this is all local.

#Step 2: Lightweight skill router (cheap)
#The router prompt is tiny — just enough to pick a skill.

def select_skill(user_query: str) -> str:
    query = user_query.lower()

    if "summarize" in query:
        return "summarization"
    if "contract" in query or "clause" in query:
        return "contract_analysis"
    return "document_qa"


#You can later replace this with:
# - embeddings
# - keyword rules
# - a small cheap model

#Step 3: Build the prompt dynamically (THIS is the savings)
#Only one skill’s instructions are injected.

from openai import OpenAI
client = OpenAI()

def run_request(user_query: str, document_text: str):
    skill_name = select_skill(user_query)
    skill = SKILLS[skill_name]

    response = client.responses.create(
        model="gpt-4.1-mini",  # or whatever you choose
        input=[
            {
                "role": "system",
                "content": skill["system_prompt"]
            },
            {
                "role": "user",
                "content": f"""
DOCUMENT:
{document_text}

QUESTION:
{user_query}
"""
            }
        ],
    )

    return response.output_text

#Token comparison (why this matters)
# Naive approach
#System:
#- QA rules (500 tokens)
#- Contract rules (600 tokens)
#- Summary rules (400 tokens)
#- Safety rules (300 tokens)
#TOTAL ≈ 1,800 tokens EVERY request

#✅ Skill-based approach
#System:
#- Selected skill only (400–600 tokens)
#At scale, this is real money saved.
