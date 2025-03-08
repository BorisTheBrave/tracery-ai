import json
import argparse
import os
from llama_cpp import Llama, LlamaGrammar
import tracery_to_ebnf_gbnf

def generate_with_grammar(
    model_path,
    prompt,
    grammar_str,
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1,
    beam_width=1,
    samples=1,
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
        beam_width: Width for beam search (1 = greedy/sampling)
        samples: Number of samples to generate
        verbose: Whether to print verbose output
    
    Returns:
        List of generated texts constrained by the grammar
    """
    # Initialize the model
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1  # Use all available GPU layers
    )
    
    # Create grammar object
    grammar = LlamaGrammar.from_string(grammar_str)
    
    results = []
    
    for sample_idx in range(samples):
        if verbose and samples > 1:
            print(f"\nGenerating sample {sample_idx+1}/{samples}...")
            
        # Generate text with grammar constraints
        if beam_width <= 1:
            # Use standard sampling if beam_width is 1 or less
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
            
            results.append(output["choices"][0]["text"])
        else:
            # Use beam search if beam_width > 1
            beams = [(prompt, 0.0)]  # (text, score) pairs
            
            for _ in range(max_tokens):
                candidates = []
                
                # For each current beam
                for beam_text, beam_score in beams:
                    # Get next token predictions
                    output = llm.create_completion(
                        beam_text,
                        max_tokens=1,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        grammar=grammar,
                        n_probs=beam_width  # Get probabilities for top tokens
                    )
                    
                    if verbose and _ == 0 and sample_idx == 0:
                        print(f"Beam search iteration 0, candidates:")
                        print(json.dumps(output, indent=2))
                    
                    # Get the generated token and its log probability
                    for choice in output["choices"]:
                        if "logprobs" in choice and choice["logprobs"]["top_logprobs"]:
                            for token, logprob in choice["logprobs"]["top_logprobs"][0].items():
                                new_text = beam_text + token
                                new_score = beam_score + logprob
                                candidates.append((new_text, new_score))
                        else:
                            # Fallback if logprobs not available
                            new_text = beam_text + choice["text"]
                            candidates.append((new_text, beam_score))
                
                # Sort candidates by score and keep top beam_width
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]
                
                if verbose and _ % 10 == 0 and sample_idx == 0:
                    print(f"Beam search iteration {_}, top beam: {beams[0][0]}")
                
                # Check if all beams end with a terminal token
                if all(llm.tokenize(beam[0])[-1] == llm.token_eos() for beam in beams):
                    break
            
            # Return the highest scoring beam
            best_text = beams[0][0][len(prompt):]
            
            if verbose:
                print(f"Final beams for sample {sample_idx+1}:")
                for i, (text, score) in enumerate(beams):
                    print(f"Beam {i}, score: {score}")
                    print(text[len(prompt):])
                    print("-" * 40)
            
            results.append(best_text)
    
    return results

def load_grammar(grammar_path, root_rule=None):
    """
    Load grammar from file, automatically detecting format (GBNF or Tracery JSON).
    
    Args:
        grammar_path: Path to the grammar file
        root_rule: Root rule name for Tracery conversion (if applicable)
        
    Returns:
        Grammar string in GBNF format
    """
    with open(grammar_path, "r") as f:
        content = f.read().strip()
    
    # Try to determine if it's JSON (Tracery) or GBNF
    is_tracery = False
    try:
        # Check if it's valid JSON
        tracery_grammar = json.loads(content)
        # If we can parse it as JSON and it's a dictionary, assume it's Tracery
        if isinstance(tracery_grammar, dict):
            is_tracery = True
    except json.JSONDecodeError:
        # Not valid JSON, assume it's GBNF
        is_tracery = False
    
    if not is_tracery:
        # It's a GBNF grammar, return as is
        return content
    else:
        # It's a Tracery grammar, convert to GBNF
        
        # Convert to GBNF
        if not root_rule:
            root_rule = next(iter(tracery_grammar))
        
        # Modify the grammar to ensure the specified root rule is used as the root
        if root_rule != "root" and "root" not in tracery_grammar:
            # Create a temporary copy of the grammar with a root rule
            temp_grammar = tracery_grammar.copy()
            temp_grammar["root"] = [f"#{root_rule}#"]
            gbnf_grammar = tracery_to_ebnf_gbnf.tracery_to_gbnf(temp_grammar)
        else:
            gbnf_grammar = tracery_to_ebnf_gbnf.tracery_to_gbnf(tracery_grammar)
        
        if os.environ.get("DEBUG"):
            print(f"Converted Tracery grammar to GBNF:\n{gbnf_grammar}")
            
        return gbnf_grammar

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text using a Llama model with grammar constraints.")
    
    parser.add_argument("--model", "-m", type=str, default="models/qwen2-0_5b-instruct-q8_0.gguf",
                        help="Path to the Llama model file")
    
    parser.add_argument("--grammar", "-g", type=str, default="example.gbnf",
                        help="Path to the grammar file (GBNF or Tracery JSON, auto-detected)")
    
    parser.add_argument("--root", type=str, default=None,
                        help="Root rule name for Tracery grammar conversion (default: first rule in grammar)")
    
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
    
    parser.add_argument("--beam-width", "-b", type=int, default=1,
                        help="Beam width for beam search (1 = greedy/sampling)")
    
    parser.add_argument("--samples", "-s", type=int, default=1,
                        help="Number of samples to generate")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output including full model response")
    
    return parser.parse_args()

# Example: Generate a JSON response
if __name__ == "__main__":
    args = parse_arguments()
    
    # Load grammar from file, auto-detecting format
    grammar = load_grammar(args.grammar, root_rule=args.root)
    
    generated_texts = generate_with_grammar(
        model_path=args.model,
        prompt=args.prompt,
        grammar_str=grammar,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        beam_width=args.beam_width,
        samples=args.samples,
        verbose=args.verbose
    )
    
    # Print all generated samples
    for i, text in enumerate(generated_texts):
        if args.samples > 1:
            print(f"\n--- Sample {i+1}/{args.samples} ---")
        print(text)
    
