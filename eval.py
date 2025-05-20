import argparse
import json
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import BaseModel
from switchai import SwitchAI
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
        default="ollama",
        help="Provider of the model to evaluate.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="llama3.2",
        help="Name of the model to evaluate.",
    )

    args = parser.parse_args()

    client = SwitchAI(provider=args.provider, model_name=args.model_name)

    if args.responses_file == "":
        args.responses_file = f"responses_{args.provider}_{args.model_name}.jsonl"
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
                    response = client.chat(
                        entry["context"], response_format=EmotionResponse
                    )
                    responses[entry["id"]] = response.message.content
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response.message.content,
                                }
                            )
                            + "\n"
                        )

                response = responses[entry["id"]]

                try:
                    response = EmotionResponse.model_validate_json(response)
                    results["emotion"].append(
                        response.emotion == entry["checks"][0]["args"]
                    )
                except pydantic.ValidationError:
                    results["emotion"].append(False)

            if entry["type"] == "decision":
                if not entry["id"] in responses:
                    response = client.chat(
                        entry["context"], response_format=DecisionResponse
                    )
                    responses[entry["id"]] = response.message.content
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response.message.content,
                                }
                            )
                            + "\n"
                        )

                response = responses[entry["id"]]

                try:
                    response = DecisionResponse.model_validate_json(response)
                    results["decision"].append(
                        response.decision == entry["checks"][0]["args"]
                    )
                except pydantic.ValidationError:
                    results["decision"].append(False)

            if entry["type"] == "in‐character":
                if not entry["id"] in responses:
                    response = client.chat(entry["context"])
                    responses[entry["id"]] = response.message.content
                    with open(args.responses_file, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": entry["id"],
                                    "response": response.message.content,
                                }
                            )
                            + "\n"
                        )

                response = responses[entry["id"]]

                results["in‐character"].append(
                    not check_contains_keywords(response, entry["checks"][0]["args"])
                )

    in_character_avg = sum(results["in‐character"]) / len(results["in‐character"])
    decision_avg = sum(results["decision"]) / len(results["decision"])
    emotion_avg = sum(results["emotion"]) / len(results["emotion"])
    average_of_averages = (in_character_avg + decision_avg + emotion_avg) / 3

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
    overall_count = (
        len(results["in‐character"])
        + len(results["decision"])
        + len(results["emotion"])
    )
    print(f"{'Overall Average':<50}{average_of_averages:<15.4f}{overall_count:<10}")
