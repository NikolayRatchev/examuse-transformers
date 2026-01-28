import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "acl-submission-anonym/EAM-spectral"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # IMPORTANT: very explicit seed
    seed = "<bos> <time_1>"
    inputs = tokenizer(seed, return_tensors="pt")

    print("Input IDs:", inputs["input_ids"])

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_time_length=[32],   # REQUIRED by EAM
            do_sample=False         # deterministic, safer
        )

    print("Raw output IDs:")
    print(outputs)

    print("\nDecoded WITHOUT skipping special tokens:")
    decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(decoded_full)

    print("\nDecoded WITH skipping special tokens:")
    decoded_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_clean)

if __name__ == "__main__":
    main()
