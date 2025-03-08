import json
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

grammar = open("example.gbnf", "r").read()

# Example: Generate a JSON response
if __name__ == "__main__":
    # Replace with your model path
    MODEL_PATH = "models/qwen2-0_5b-instruct-q8_0.gguf"
    
    prompt = " "
    
    generated_text = generate_with_grammar(
        model_path=MODEL_PATH,
        prompt=prompt,
        grammar_str=grammar,
        verbose=False
    )
    
    print(generated_text)
    
