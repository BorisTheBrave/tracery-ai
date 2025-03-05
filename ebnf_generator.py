#!/usr/bin/env python3
"""
Generates strings matching a given EBNF grammar using the outlines library.
Works with EBNF grammars produced by tracery_to_ebnf.py.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional

import outlines


def generate_from_grammar(
    grammar_text: str, 
    model: Any, 
    num_samples: int = 1, 
    temperature: float = 0.7,
    max_tokens: int = 100
) -> list[str]:
    """
    Generate strings that match the given EBNF grammar.
    
    Args:
        grammar_text: EBNF grammar as a string
        model: Language model to use for generation
        num_samples: Number of samples to generate
        temperature: Temperature for generation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        List of generated strings
    """
    sampler = outlines.samplers.multinomial(num_samples)

    generator = outlines.generate.cfg(model, grammar_text, sampler)

    return generator(" ")

DEFAULT_MODEL = "gpt2"

def main():
    parser = argparse.ArgumentParser(description="Generate text from an EBNF grammar using outlines.")
    parser.add_argument("grammar_file", help="Path to the EBNF grammar file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate (default: 100)")
    parser.add_argument("--output", help="Output file for generated samples (default: print to stdout)")
    
    args = parser.parse_args()
    
    try:
        # Load the grammar
        with open(args.grammar_file, 'r') as f:
            grammar_text = f.read()
        
        # Load the model
        try:
            model = model = outlines.models.transformers(args.model)
        except Exception as e:
            print(f"Error loading model '{args.model}': {e}")
            sys.exit(1)
        
        # Generate samples
        samples = generate_from_grammar(
            grammar_text, 
            model, 
            num_samples=args.samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Output the samples
        if args.output:
            with open(args.output, 'w') as f:
                for i, sample in enumerate(samples, 1):
                    f.write(f"Sample {i}:\n{sample}\n\n")
            print(f"Generated {len(samples)} samples to {args.output}")
        else:
            for i, sample in enumerate(samples, 1):
                print(f"Sample {i}:\n{sample}\n")
        
    except FileNotFoundError:
        print(f"Error: File '{args.grammar_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main() 