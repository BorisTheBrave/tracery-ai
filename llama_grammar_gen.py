import json
import argparse
from llama_cpp import Llama, LlamaGrammar

def generate_with_grammar(
    model_path,
    prompt,
    grammar_str,
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1,
    verbose=False
):
    """
    Generate text using a Llama model with grammar constraints.
    
    Args:
        model_path: Path to the Llama model file
        prompt: Input prompt for generation
        grammar_str: Grammar string in GBNF format
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repeat_penalty: Penalty for repeating tokens
        verbose: Whether to print verbose output
    
    Returns:
        Generated text constrained by the grammar
    """
    # Initialize the model
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1  # Use all available GPU layers
    )
    
    # Create grammar object
    grammar = LlamaGrammar.from_string(grammar_str)
    
    # Generate text with grammar constraints
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        grammar=grammar
    )
    
    if verbose:
        print(json.dumps(output, indent=2))
    
    return output["choices"][0]["text"]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text using a Llama model with grammar constraints.")
    
    parser.add_argument("--model", "-m", type=str, default="models/qwen2-0_5b-instruct-q8_0.gguf",
                        help="Path to the Llama model file")
    
    parser.add_argument("--grammar", "-g", type=str, default="example.gbnf",
                        help="Path to the grammar file in GBNF format")
    
    parser.add_argument("--prompt", "-p", type=str, default=" ",
                        help="Input prompt for generation")
    
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature (higher = more creative, lower = more deterministic)")
    
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    
    parser.add_argument("--repeat-penalty", type=float, default=1.1,
                        help="Penalty for repeating tokens")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output including full model response")
    
    return parser.parse_args()

# Example: Generate a JSON response
if __name__ == "__main__":
    args = parse_arguments()
    
    # Load grammar from file
    with open(args.grammar, "r") as f:
        grammar = f.read()
    
    generated_text = generate_with_grammar(
        model_path=args.model,
        prompt=args.prompt,
        grammar_str=grammar,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        verbose=args.verbose
    )
    
    print(generated_text)
    
