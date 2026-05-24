# =============================================================================
# HANDS-ON EXERCISE: Function Calling with OpenAI
# =============================================================================
# What you're building:
#   A Claim Fact-Checker bot with TWO tools:
#     1. search_knowledge_base(claim)  — finds similar FDA violation examples
#     2. check_fda_status(drug_name)   — looks up a drug's regulatory history
#
# The LLM decides which tool(s) to call, in what order.
# You run the tools. The LLM synthesizes the final answer.
#
# Run with: python [C] function_calling_exercise.py
# Requires: pip install openai
# Set your key: export OPENAI_API_KEY="sk-..."
# =============================================================================

import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# =============================================================================
# STAGE 1 — THE MOCK TOOLS (your "real world")
# =============================================================================
# In production these would call a real DB, vector store, or API.
# For this exercise they return hardcoded data so you can run it instantly.

def search_knowledge_base(claim: str) -> dict:
    """
    Simulates a hybrid retrieval search over FDA violation examples.
    In HybridRAG this is your dense + sparse + RRF fusion step.
    """
    # Fake KB — pretend these came from your pgvector store
    examples = [
        {
            "claim": "Drug X eliminates all cancer cells within 30 days",
            "label": "misleading_claim",
            "source": "FDA Untitled Letter, 2023-04-12",
            "rationale": "Absolute language ('eliminates all') with no supporting trial data cited."
        },
        {
            "claim": "Drug X showed 40% tumor reduction in Phase II trial",
            "label": "supported_claim",
            "source": "FDA Review NDA-214087, 2022-11-01",
            "rationale": "Claim is consistent with published Phase II results."
        },
        {
            "claim": "Drug X is the only FDA-approved treatment for condition Y",
            "label": "misleading_claim",
            "source": "FDA Warning Letter, 2023-07-19",
            "rationale": "Three other approved treatments exist for condition Y."
        },
    ]
    return {
        "query": claim,
        "top_matches": examples[:2],   # return top 2
        "retrieval_score": 0.82
    }


def check_fda_status(drug_name: str) -> dict:
    """
    Simulates an FDA regulatory status lookup for a drug.
    In production this would query the FDA's openFDA API.
    """
    # Fake regulatory records
    registry = {
        "Drug X": {
            "approval_status": "approved",
            "approved_indications": ["condition Y (adjunct therapy)"],
            "warning_letters": 2,
            "last_action": "Untitled Letter issued 2023-04-12 for promotional material violations",
            "notes": "Not approved as monotherapy or for cancer cure claims."
        },
        "Drug Y": {
            "approval_status": "approved",
            "approved_indications": ["condition Z"],
            "warning_letters": 0,
            "last_action": "None",
            "notes": "Clean record."
        }
    }
    # Default if drug not found
    return registry.get(drug_name, {
        "approval_status": "not_found",
        "warning_letters": 0,
        "notes": f"No record found for '{drug_name}' in mock registry."
    })


# =============================================================================
# STAGE 2 — THE TOOL SCHEMAS (what the LLM sees)
# =============================================================================
# This is the "menu" you hand to the LLM.
# The LLM reads the descriptions to decide when and how to call each tool.
# RULE: Write descriptions like documentation for a smart but uninformed colleague.

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the FDA violation knowledge base for examples similar to a given claim. "
                "Returns the top matching examples with their labels, sources, and rationales. "
                "Use this to find precedent before classifying a claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The pharmaceutical marketing claim to search for."
                    }
                },
                "required": ["claim"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_fda_status",
            "description": (
                "Look up a drug's FDA approval status and regulatory history, including "
                "any warning letters or untitled letters issued. Use this when the claim "
                "mentions a specific drug name and you need to verify its approved indications."
            ),
            "parameters": {
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
    }
]


# =============================================================================
# STAGE 3 — THE AGENT LOOP
# =============================================================================
# This is the core pattern. Study this loop — it's the same loop
# used in every function-calling agent, from simple bots to AutoGPT.

def run_agent(user_query: str) -> str:
    """
    Runs a single-turn function-calling agent.
    Handles one or multiple tool calls before giving a final answer.
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")

    # Dispatcher: maps tool names to actual Python functions
    available_functions = {
        "search_knowledge_base": search_knowledge_base,
        "check_fda_status": check_fda_status,
    }

    # Start the conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a pharmaceutical claim compliance analyst. "
                "Use the available tools to look up relevant FDA examples and regulatory "
                "status before classifying a claim. Be specific about why a claim is "
                "misleading or supported, citing the tool results."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # -------------------------------------------------------------------------
    # ROUND 1: Send to LLM — it may respond with tool calls
    # -------------------------------------------------------------------------
    print("\n[Round 1] Sending to LLM...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"   # LLM decides whether to use tools
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # If no tool calls, LLM answered directly (e.g. simple clarification question)
    if not tool_calls:
        print("[No tool calls made — LLM answered directly]")
        return response_message.content

    # -------------------------------------------------------------------------
    # EXECUTE ALL TOOL CALLS
    # -------------------------------------------------------------------------
    # Add the LLM's tool-call request to message history first
    messages.append(response_message)

    print(f"\n[LLM requested {len(tool_calls)} tool call(s)]")

    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)  # ← always json.loads!

        print(f"\n  → Calling: {fn_name}({fn_args})")

        # Look up and run the actual function
        fn_to_call = available_functions[fn_name]
        result = fn_to_call(**fn_args)

        print(f"  ← Result: {json.dumps(result, indent=2)[:200]}...")  # preview

        # Add the tool result to message history
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,   # ties result to the request
            "content": json.dumps(result)
        })

    # -------------------------------------------------------------------------
    # ROUND 2: Send results back to LLM for final answer
    # -------------------------------------------------------------------------
    print("\n[Round 2] Sending tool results back to LLM...")
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    final_answer = final_response.choices[0].message.content
    print(f"\nASSISTANT: {final_answer}")
    return final_answer


# =============================================================================
# STAGE 4 — RUN IT
# =============================================================================
# Try different queries and observe which tools the LLM decides to call.
# Change the query and re-run to build intuition for how the LLM routes.

if __name__ == "__main__":

    # --- Query 1: Vague claim — LLM should search KB + check FDA ---
    run_agent(
        "Is the claim 'Drug X eliminates all cancer cells within 30 days' acceptable "
        "for use in promotional materials?"
    )

    # --- Uncomment to try more queries ---

    # Query 2: Only needs KB search (no specific drug named)
    # run_agent(
    #     "What kind of language typically gets flagged in FDA warning letters?"
    # )

    # Query 3: Only needs FDA status check
    # run_agent(
    #     "Is Drug X FDA-approved as a cancer monotherapy?"
    # )

    # Query 4: Watch the LLM decide NOT to call any tool
    # run_agent(
    #     "What does 'misleading claim' mean in the context of pharmaceutical advertising?"
    # )
