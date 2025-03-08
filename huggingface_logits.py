import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load a model from Hugging Face and get logits")
    parser.add_argument("--model", type=str, required=True, help="Model name or path on Hugging Face")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to get logits for")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the model on (cuda/cpu)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top logits to display")
    return parser.parse_args()

def get_logits(model_name, prompt, device="cuda", top_k=10):
    print(f"Loading model {model_name} on {device}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get model output with logits
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    
    # Get logits from the last token
    logits = outputs.logits[0, -1, :]
    
    # Get top-k logits and their corresponding tokens
    topk_values, topk_indices = torch.topk(logits, top_k)
    
    # Print results
    print(f"\nTop {top_k} logits for prompt: '{prompt}'")
    print("-" * 50)
    for i, (value, idx) in enumerate(zip(topk_values.cpu().numpy(), topk_indices.cpu().numpy())):
        token = tokenizer.decode([idx])
        print(f"{i+1}. Token: '{token}', Index: {idx}, Logit: {value:.4f}")

    if prompt.endswith(" "):
        fruits = ["apple", "banana", "orange"]
    else:
        fruits = [" apple", " banana", " orange"]

    print()
    for fruit in fruits:
        tokens = tokenizer.encode(fruit)
        print(f"Logits for \"{tokenizer.decode(tokens[0])}\": {logits[tokens[0]]}")

    return logits

def main():
    args = parse_arguments()
    print(get_logits(args.model, args.prompt, args.device, args.top_k))

if __name__ == "__main__":
    main() 