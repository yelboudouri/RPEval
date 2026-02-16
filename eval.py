import argparse
import json
import os
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm


class EmotionResponse(BaseModel):
    emotion: Literal[
        "determination",
        "anger",
        "anticipation",
        "excitement",
        "sadness",
        "disgust",
        "fear",
        "surprise",
        "joy",
        "hope",
        "love",
        "guilt",
        "pride",
    ]
    text: str


class DecisionResponse(BaseModel):
    decision: Literal["yes", "no"]


def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text


def check_contains_keywords(text, keywords):
    return any(normalize_text(keyword) in normalize_text(text) for keyword in keywords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPEval")
    parser.add_argument(
        "--responses-file",
        type=str,
        default="",
        help="Path to responses file. Used as a cache to continue evaluation if interrupted or to recompute results without querying the API.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        help="Provider of the model to evaluate.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/llama-3.2-3b-instruct",
        help="Name of the model to evaluate (e.g., meta-llama/llama-3.2-3b-instruct for OpenRouter).",
    )

    args = parser.parse_args()

    # Initialize OpenRouter Client via OpenAI SDK
    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    if args.responses_file == "":
        args.responses_file = f"responses_{args.provider}_{args.model_name.replace('/', '_')}.jsonl"
    file_path = Path(args.responses_file)
    file_path.touch(exist_ok=True)

    results = {
        "in‐character": [],
        "decision": [],
        "emotion": [],
    }
    responses = {}
    with open(args.responses_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            responses[entry["id"]] = entry["response"]

    with open("data/eval_data.jsonl", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Evaluating..", total=len(lines)):
            entry = json.loads(line)

            if entry["type"] == "emotion":
                if not entry["id"] in responses:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": entry["context"]}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "emotion_response",
                                "schema": EmotionResponse.model_json_schema(),
                                "strict": True
                            }
                        }
                    )
                    response_text = response.choices[0].message.content
                    responses[entry["id"]] = response_text
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response_text,
                                }
                            )
                            + "\n"
                        )

                response_text = responses[entry["id"]]

                try:
                    parsed_response = EmotionResponse.model_validate_json(response_text)
                    results["emotion"].append(
                        parsed_response.emotion == entry["checks"][0]["args"][0]
                    )
                except pydantic.ValidationError:
                    results["emotion"].append(False)

            if entry["type"] == "decision":
                if not entry["id"] in responses:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": entry["context"]}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "decision_response",
                                "schema": DecisionResponse.model_json_schema(),
                                "strict": True
                            }
                        }
                    )
                    response_text = response.choices[0].message.content
                    responses[entry["id"]] = response_text
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response_text,
                                }
                            )
                            + "\n"
                        )

                response_text = responses[entry["id"]]

                try:
                    parsed_response = DecisionResponse.model_validate_json(response_text)
                    results["decision"].append(
                        parsed_response.decision == entry["checks"][0]["args"][0]
                    )
                except pydantic.ValidationError:
                    results["decision"].append(False)

            if entry["type"] == "in‐character":
                if not entry["id"] in responses:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": entry["context"]}]
                    )
                    response_text = response.choices[0].message.content
                    responses[entry["id"]] = response_text
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response_text,
                                }
                            )
                            + "\n"
                        )

                response_text = responses[entry["id"]]

                results["in‐character"].append(
                    not check_contains_keywords(response_text, entry["checks"][0]["args"])
                )

    in_character_avg = sum(results["in‐character"]) / len(results["in‐character"]) if results["in‐character"] else 0
    decision_avg = sum(results["decision"]) / len(results["decision"]) if results["decision"] else 0
    emotion_avg = sum(results["emotion"]) / len(results["emotion"]) if results["emotion"] else 0
    
    overall_count = (
        len(results["in‐character"])
        + len(results["decision"])
        + len(results["emotion"])
    )
    average_of_averages = (in_character_avg + decision_avg + emotion_avg) / 3 if overall_count > 0 else 0

    # Print report
    print(f"{'Category':<50}{'Average':<15}{'Count':<10}")
    print("-" * 70)
    print(
        f"{'In‐character Consistency':<50}{in_character_avg:<15.4f}{len(results['in‐character']):<10}"
    )
    print(
        f"{'Decision-making & Moral Alignment':<50}{decision_avg:<15.4f}{len(results['decision']):<10}"
    )
    print(
        f"{'Emotional Understanding':<50}{emotion_avg:<15.4f}{len(results['emotion']):<10}"
    )
    print("-" * 70)
    print(f"{'Overall Average':<50}{average_of_averages:<15.4f}{overall_count:<10}")
