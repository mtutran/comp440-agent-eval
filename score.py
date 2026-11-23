import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

AUTHOR = "Tu Tran"

INSTRUCTIONS = """You are grading how well a customer support agent handled a customer.
Score the agent on each criterion of the rubric below, in order. For each criterion, give one
sentence of evidence from the conversation, then a whole-number score from 1 to 5.

"""


class Score(BaseModel):
    reason: str
    score: Literal[1, 2, 3, 4, 5]


class Scores(BaseModel):
    criterion_1: Score
    criterion_2: Score
    criterion_3: Score


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="score only the first N conversations")
args = parser.parse_args()

load_dotenv(override=True)
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("No API key found: paste your key into .env as ANTHROPIC_API_KEY=...")

rubric = open("rubric.md").read()
criteria = [line[3:].strip() for line in rubric.splitlines() if line.startswith("## ")]
if len(criteria) != 3:
    sys.exit(f"rubric.md must have exactly three '## ' criteria, but it has {len(criteria)}.")

conversations = json.load(open("data/conversations.json"))
if args.limit:
    conversations = conversations[: args.limit]

client = anthropic.Anthropic(max_retries=5)


def score_one(conversation):
    try:
        response = client.messages.parse(
            model="claude-sonnet-5-5",
            max_tokens=2000,
            system=INSTRUCTIONS + rubric,
            messages=[{"role": "user", "content": conversation["conversation"]}],
            output_format=Scores,
        )
        s = response.parsed_output
        parts = [s.criterion_1, s.criterion_2, s.criterion_3]
    except Exception as e:
        print(f"Skipped conversation {conversation['id']}: {e}")
        return None
    return {
        "conversation_id": conversation["id"],
        "overall": round(sum(p.score for p in parts) / 3, 2),
        "scores": [
            {"criterion": name, "score": p.score, "reason": p.reason}
            for name, p in zip(criteria, parts)
        ],
    }


with ThreadPoolExecutor(max_workers=5) as pool:
    results = [r for r in pool.map(score_one, conversations) if r]

with open("results.json", "w") as f:
    json.dump({"author": AUTHOR, "results": results}, f, indent=2)

print(f"Scored {len(results)} conversations.")
if results:
    print(f"Average overall score: {sum(r['overall'] for r in results) / len(results):.2f}")
