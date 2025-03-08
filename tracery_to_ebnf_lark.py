#!/usr/bin/env python3
"""
Converts a Tracery grammar to Lark-style EBNF notation.
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
    
    def to_ebnf(self) -> str:
        """Convert to EBNF format."""
        escaped_text = self.text.replace('"', '\\"')
        return f'"{escaped_text}"'


class SymbolToken(Token):
    """A symbol reference token (#symbol#)."""
    def __init__(self, symbol: str):
        self.symbol = symbol
    
    def to_ebnf(self) -> str:
        """Convert to EBNF format."""
        return self.symbol


class VariableToken(Token):
    """A variable assignment token (#[var:val]#)."""
    def __init__(self, variable: str, value: str):
        self.variable = variable
        self.value = value
    
    def to_ebnf(self) -> str:
        """
        Variables are not supported in EBNF conversion.
        This method should not be called.
        """
        raise NotImplementedError("Variable tokens cannot be converted to EBNF")


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
    Convert a list of tokens to a Lark-style EBNF expression.
    
    Args:
        tokens: List of Token objects
        
    Returns:
        EBNF expression
    """
    if not tokens:
        return '""'  # Empty rule
    
    formatted_tokens = [token.to_ebnf() for token in tokens]
    return " ".join(formatted_tokens)


def tracery_to_ebnf(grammar: Dict[str, List[str]]) -> str:
    """
    Convert a Tracery grammar to Lark-style EBNF notation.
    """
    ebnf = []
    grammar_keys = list(grammar.keys())
    
    # Add a start rule if "origin" exists in the grammar
    if "origin" in grammar_keys:
        ebnf.append("?start: origin")
        ebnf.append("")
    
    # Process each symbol in the grammar
    for symbol, expansions in grammar.items():
        # Format the rule name
        rule_name = symbol
        
        # Format the expansions
        formatted_expansions = []
        for i, expansion in enumerate(expansions):
            if isinstance(expansion, str):
                try:
                    # Parse the rule into tokens
                    tokens = parse_rule(expansion, grammar_keys)
                    
                    # Convert tokens to EBNF
                    formatted_expansion = convert_rule(tokens)
                    
                    # Add a rule name for each expansion (except the first one)
                    if i > 0:
                        formatted_expansion += f"  -> {symbol}_{i}"
                    
                    formatted_expansions.append(formatted_expansion)
                except ValueError as e:
                    # Re-raise with more context
                    raise ValueError(f"Error in rule '{symbol}': {e}")
        
        # Join the expansions with newlines and proper indentation
        if len(formatted_expansions) > 1:
            rule_body = "\n| ".join(formatted_expansions)
            ebnf_rule = f"?{rule_name}: {formatted_expansions[0]}"
            for i, exp in enumerate(formatted_expansions[1:], 1):
                ebnf_rule += f"\n| {exp}"
        else:
            rule_body = formatted_expansions[0] if formatted_expansions else '""'
            ebnf_rule = f"{rule_name}: {rule_body}"
        
        ebnf.append(ebnf_rule)
        ebnf.append("")  # Add a blank line between rules
    
    # Add common imports
    ebnf.append("%import common.WS")
    ebnf.append("%ignore WS")
    
    return "\n".join(ebnf)


def main():
    if len(sys.argv) != 2:
        print("Usage: python tracery_to_ebnf.py <tracery_grammar_file.json>")
        sys.exit(1)
    
    grammar_file = sys.argv[1]
    
    try:
        with open(grammar_file, 'r') as f:
            grammar = json.load(f)
        
        ebnf = tracery_to_ebnf(grammar)
        print(ebnf)
        
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