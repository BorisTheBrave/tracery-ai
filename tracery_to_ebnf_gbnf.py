#!/usr/bin/env python3
"""
Converts a Tracery grammar to GBNF (GGML BNF) notation.
Fails if the grammar contains any variables (e.g., #[variable:value]#).
Symbol references like #symbol# are allowed and converted properly.
"""

import json
import re
import sys
from typing import Dict, List, Any, Union


class Token:
    """Base class for all tokens in a Tracery rule."""
    pass


class LiteralToken(Token):
    """A literal text token."""
    def __init__(self, text: str):
        self.text = text
    
    def to_gbnf(self) -> str:
        """Convert to GBNF format."""
        escaped_text = self.text.replace('"', '\\"')
        return f'"{escaped_text}"'


class SymbolToken(Token):
    """A symbol reference token (#symbol#)."""
    def __init__(self, symbol: str):
        self.symbol = symbol
    
    def to_gbnf(self) -> str:
        """Convert to GBNF format."""
        return self.symbol


class VariableToken(Token):
    """A variable assignment token (#[var:val]#)."""
    def __init__(self, variable: str, value: str):
        self.variable = variable
        self.value = value
    
    def to_gbnf(self) -> str:
        """
        Variables are not supported in GBNF conversion.
        This method should not be called.
        """
        raise NotImplementedError("Variable tokens cannot be converted to GBNF")


def parse_rule(rule: str, grammar_keys: List[str]) -> List[Token]:
    """
    Parse a Tracery rule into a list of tokens.
    
    Args:
        rule: The Tracery rule to parse
        grammar_keys: List of valid symbol names in the grammar
        
    Returns:
        List of Token objects
    """
    # Check for variable assignments (not supported)
    var_matches = re.finditer(r'#\[([^:]+):([^\]]+)\]#', rule)
    for match in var_matches:
        var_name = match.group(1)
        var_value = match.group(2)
        raise ValueError(f"Variable assignment found: #{var_name}:{var_value}#")
    
    # Split the rule into tokens
    tokens = []
    current_text = ""
    i = 0
    
    while i < len(rule):
        if rule[i:i+1] == "#":
            # Possible symbol reference
            if current_text:
                tokens.append(LiteralToken(current_text))
                current_text = ""
            
            # Find the closing #
            end = rule.find("#", i+1)
            if end == -1:
                # No closing #, treat as literal
                current_text += rule[i]
                i += 1
                continue
            
            symbol_name = rule[i+1:end]
            # Check if it's a variable assignment
            if symbol_name.startswith("[") and "]" in symbol_name:
                # This should have been caught earlier, but just in case
                raise ValueError(f"Variable assignment found: #{symbol_name}#")
            
            if symbol_name in grammar_keys:
                tokens.append(SymbolToken(symbol_name))
            else:
                # Not a valid symbol, treat as literal
                tokens.append(LiteralToken(f"#{symbol_name}#"))
            
            i = end + 1
        else:
            current_text += rule[i]
            i += 1
    
    if current_text:
        tokens.append(LiteralToken(current_text))
    
    return tokens


def convert_rule(tokens: List[Token]) -> str:
    """
    Convert a list of tokens to a GBNF expression.
    
    Args:
        tokens: List of Token objects
        
    Returns:
        GBNF expression
    """
    if not tokens:
        return '""'  # Empty rule
    
    formatted_tokens = [token.to_gbnf() for token in tokens]
    return " ".join(formatted_tokens)


def tracery_to_gbnf(grammar: Dict[str, List[str]]) -> str:
    """
    Convert a Tracery grammar to GBNF notation.
    """
    gbnf = []
    grammar_keys = list(grammar.keys())
    
    # Add a root rule if "origin" exists in the grammar
    if "origin" in grammar_keys:
        gbnf.append("root ::= origin")
        gbnf.append("")
    
    # Process each symbol in the grammar
    for symbol, expansions in grammar.items():
        # Format the rule name
        rule_name = symbol
        
        # Format the expansions
        formatted_expansions = []
        for expansion in expansions:
            if isinstance(expansion, str):
                try:
                    # Parse the rule into tokens
                    tokens = parse_rule(expansion, grammar_keys)
                    
                    # Convert tokens to GBNF
                    formatted_expansion = convert_rule(tokens)
                    formatted_expansions.append(formatted_expansion)
                except ValueError as e:
                    # Re-raise with more context
                    raise ValueError(f"Error in rule '{symbol}': {e}")
        
        # Join the expansions with | for alternatives
        rule_body = " | ".join(formatted_expansions) if formatted_expansions else '""'
        gbnf_rule = f"{rule_name} ::= {rule_body}"
        
        gbnf.append(gbnf_rule)
        gbnf.append("")  # Add a blank line between rules
    
    # Add whitespace rule
    gbnf.append("ws ::= [ \\t\\n]+")
    
    return "\n".join(gbnf)


def main():
    if len(sys.argv) != 2:
        print("Usage: python tracery_to_gbnf.py <tracery_grammar_file.json>")
        sys.exit(1)
    
    grammar_file = sys.argv[1]
    
    try:
        with open(grammar_file, 'r') as f:
            grammar = json.load(f)
        
        gbnf = tracery_to_gbnf(grammar)
        print(gbnf)
        
    except FileNotFoundError:
        print(f"Error: File '{grammar_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{grammar_file}' is not a valid JSON file.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 