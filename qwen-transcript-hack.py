from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    model_name = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    input_text = (
        "This is a transcript of a conversation between a helpful bot, 'Bot', "
        "and a human, 'User'.  The bot is very intelligent and always answers "
        "the human's questions with a useful reply.\n\n"
        "User: Provide a synonym for 'bright'\n\n"
        "Bot: "
    )
    out = pipe(
        input_text,
        max_new_tokens=80,
        do_sample=True,      # set False for greedy
        temperature=0.7,
        top_p=0.9
    )

    print(out[0]["generated_text"])


if __name__ == "__main__":
    main()
