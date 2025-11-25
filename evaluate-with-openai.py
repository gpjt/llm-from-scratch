import json

from openai import OpenAI
from tqdm import tqdm


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


def query_model(prompt):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )

    return response.output_text


def generate_model_scores(json_data):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}` "
            f"on a scale of 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


def main():
    file_path = "instruction-data-with-responses.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)

    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}` "
            f"on a scale of 0 to 100, where 100 is the best score."
        )
        print("\nDataset response:")
        print(">>", entry["output"])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-----------------------------------------")

    scores = generate_model_scores(test_data)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")




if __name__ == "__main__":
    main()
