# =============================================================================
# ANTHROPIC TOOL USE — same exercise, different API shape
# Compare this file against [C] function_calling_exercise.py
# Search for "# DIFF" comments to see exactly what changed (only 4 things)
# =============================================================================
# Run:  python "[C] function_calling_anthropic.py"
# Deps: pip install anthropic
# Key:  export ANTHROPIC_API_KEY="sk-ant-..."
# =============================================================================

import json
import os
import anthropic                          # DIFF 0: different SDK

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# =============================================================================
# MOCK TOOLS — identical to the Groq version, nothing changes here
# =============================================================================

def search_knowledge_base(claim: str) -> dict:
    examples = [
        {
            "claim": "Drug X eliminates all cancer cells within 30 days",
            "label": "misleading_claim",
            "source": "FDA Untitled Letter, 2023-04-12",
            "rationale": "Absolute language with no supporting trial data cited."
        },
        {
            "claim": "Drug X showed 40% tumor reduction in Phase II trial",
            "label": "supported_claim",
            "source": "FDA Review NDA-214087, 2022-11-01",
            "rationale": "Claim is consistent with published Phase II results."
        },
    ]
    return {"query": claim, "top_matches": examples[:2], "retrieval_score": 0.82}


def check_fda_status(drug_name: str) -> dict:
    registry = {
        "Drug X": {
            "approval_status": "approved",
            "approved_indications": ["condition Y (adjunct therapy)"],
            "warning_letters": 2,
            "last_action": "Untitled Letter issued 2023-04-12 for promotional material violations",
            "notes": "Not approved as monotherapy or for cancer cure claims."
        }
    }
    return registry.get(drug_name, {
        "approval_status": "not_found",
        "notes": f"No record found for '{drug_name}'."
    })


# =============================================================================
# DIFF 1 — Tool schema shape
# OpenAI:    {"type": "function", "function": {"name": ..., "parameters": {...}}}
# Anthropic: {"name": ..., "input_schema": {...}}   ← flatter, renamed key
# =============================================================================

tools = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the FDA violation knowledge base for examples similar to a given claim. "
            "Use this to find precedent before classifying a claim."
        ),
        "input_schema": {                  # ← "parameters" in OpenAI
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The pharmaceutical marketing claim to search for."
                }
            },
            "required": ["claim"]
        }
    },
    {
        "name": "check_fda_status",
        "description": (
            "Look up a drug's FDA approval status and regulatory history. "
            "Use this when a specific drug name is mentioned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The brand or generic name of the drug to look up."
                }
            },
            "required": ["drug_name"]
        }
    }
]


# =============================================================================
# THE AGENT LOOP
# =============================================================================

def run_agent(user_query: str) -> str:
    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")

    available_functions = {
        "search_knowledge_base": search_knowledge_base,
        "check_fda_status": check_fda_status,
    }

    messages = [
        {"role": "user", "content": user_query}
        # Note: system prompt goes as a separate param below, not in messages
    ]

    # -------------------------------------------------------------------------
    # DIFF 2 — API call shape
    # OpenAI:    client.chat.completions.create(model=..., messages=..., tools=...)
    # Anthropic: client.messages.create(model=..., max_tokens=..., messages=..., tools=...)
    #            max_tokens is REQUIRED — there is no default
    # -------------------------------------------------------------------------
    print("\n[Round 1] Sending to Claude...")
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,                   # ← required, no default
        system=(
            "You are a pharmaceutical claim compliance analyst. "
            "Use the available tools to look up FDA examples and regulatory status "
            "before classifying a claim."
        ),
        messages=messages,
        tools=tools
    )

    # -------------------------------------------------------------------------
    # DIFF 3 — Reading the tool call response
    # OpenAI:    response.choices[0].message.tool_calls  (list of tool_call objects)
    #            tool_call.function.arguments  → JSON STRING, must json.loads()
    #
    # Anthropic: response.content  → list of BLOCKS (text blocks + tool_use blocks)
    #            block.input       → already a DICT, no json.loads() needed
    #            check stop_reason == "tool_use" (not "tool_calls")
    # -------------------------------------------------------------------------
    if response.stop_reason != "tool_use":
        print("[No tool calls — Claude answered directly]")
        return response.content[0].text

    # Collect all tool_use blocks from the response
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    print(f"\n[Claude requested {len(tool_use_blocks)} tool call(s)]")

    # Append Claude's full response (including tool_use blocks) to messages
    messages.append({"role": "assistant", "content": response.content})

    # -------------------------------------------------------------------------
    # DIFF 4 — Sending tool results back
    # OpenAI:    {"role": "tool", "tool_call_id": ..., "content": json.dumps(result)}
    #
    # Anthropic: {"role": "user", "content": [{"type": "tool_result", ...}]}
    #            role is "user" not "tool"
    #            content is a LIST of tool_result blocks (one per tool call)
    # -------------------------------------------------------------------------
    tool_results = []

    for block in tool_use_blocks:
        fn_name = block.name
        fn_args = block.input                  # ← already a dict, no json.loads()!

        print(f"\n  → Calling: {fn_name}({fn_args})")

        fn_to_call = available_functions[fn_name]
        result = fn_to_call(**fn_args)

        print(f"  ← Result: {json.dumps(result)[:200]}...")

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,           # ties result to the request
            "content": json.dumps(result)
        })

    # Send all results back in a single user message
    messages.append({
        "role": "user",                        # ← "tool" in OpenAI, "user" in Anthropic
        "content": tool_results                # ← list of tool_result blocks
    })

    # Round 2: get the final answer
    print("\n[Round 2] Sending tool results back to Claude...")
    final_response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a pharmaceutical claim compliance analyst. "
            "Use the available tools to look up FDA examples and regulatory status "
            "before classifying a claim."
        ),
        messages=messages,
        tools=tools
    )

    final_answer = final_response.content[0].text
    print(f"\nASSISTANT: {final_answer}")
    return final_answer


# =============================================================================
# RUN IT
# =============================================================================

if __name__ == "__main__":
    run_agent(
        "Is the claim 'Drug X eliminates all cancer cells within 30 days' acceptable "
        "for use in promotional materials?"
    )
