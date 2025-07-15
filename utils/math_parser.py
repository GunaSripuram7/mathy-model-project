"""
Mathematical formula parser and tokenizer.
Handles parsing of various mathematical notation formats and converts them to tokens.
"""

import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, sin, cos, tan, exp, log, sqrt, pi, E
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class FormulaTokenizer:
    """Tokenizer for mathematical formulas"""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[NUM]': 4,  # Placeholder for numbers
            '[VAR]': 5,  # Placeholder for variables
        }
        
        # Mathematical tokens
        self.math_tokens = {
            # Operators
            '+': 10, '-': 11, '*': 12, '/': 13, '^': 14, '**': 15,
            '=': 16, '==': 17, '!=': 18, '<': 19, '>': 20, '<=': 21, '>=': 22,
            
            # Parentheses and brackets
            '(': 30, ')': 31, '[': 32, ']': 33, '{': 34, '}': 35,
            
            # Functions
            'sin': 50, 'cos': 51, 'tan': 52, 'asin': 53, 'acos': 54, 'atan': 55,
            'sinh': 56, 'cosh': 57, 'tanh': 58,
            'exp': 60, 'log': 61, 'ln': 62, 'log10': 63,
            'sqrt': 70, 'abs': 71, 'floor': 72, 'ceil': 73,
            'max': 80, 'min': 81, 'sum': 82, 'prod': 83,
            
            # Constants
            'pi': 100, 'e': 101, 'euler': 102, 'inf': 103,
            
            # Common variables
            'x': 200, 'y': 201, 'z': 202, 't': 203, 'r': 204, 'theta': 205,
            'phi': 206, 'u': 207, 'v': 208, 'w': 209, 'n': 210, 'i': 211, 'j': 212, 'k': 213,
            'a': 220, 'b': 221, 'c': 222, 'd': 223, 'f': 224, 'g': 225, 'h': 226,
            
            # Punctuation
            ',': 300, '.': 301, ';': 302, ':': 303,
        }
        
        # Combine all tokens
        self.token_to_id = {**self.special_tokens, **self.math_tokens}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Pattern for tokenization
        self.token_pattern = self._create_token_pattern()
        
        logger.info(f"Initialized tokenizer with {len(self.token_to_id)} base tokens")
    
    def _create_token_pattern(self) -> re.Pattern:
        """Create regex pattern for tokenization"""
        # Sort tokens by length (longest first) to avoid partial matches
        tokens = sorted(self.math_tokens.keys(), key=len, reverse=True)
        
        # Escape special regex characters
        escaped_tokens = [re.escape(token) for token in tokens]
        
        # Pattern for numbers (integers and floats)
        number_pattern = r'\d+\.?\d*'
        
        # Pattern for variables (single letters or letter+digits)
        variable_pattern = r'[a-zA-Z][a-zA-Z0-9]*'
        
        # Combine all patterns
        pattern = '|'.join([
            number_pattern,
            '|'.join(escaped_tokens),
            variable_pattern,
            r'\S'  # Any other non-whitespace character
        ])
        
        return re.compile(pattern)
    
    def tokenize(self, formula: str) -> List[str]:
        """Tokenize a mathematical formula into tokens"""
        # Preprocess formula
        formula = self._preprocess_formula(formula)
        
        # Find all tokens
        tokens = self.token_pattern.findall(formula)
        
        # Post-process tokens
        processed_tokens = []
        for token in tokens:
            if token.isspace():
                continue  # Skip whitespace
            elif self._is_number(token):
                processed_tokens.append('[NUM]')
            elif token in self.math_tokens:
                processed_tokens.append(token)
            elif self._is_variable(token):
                if token in self.math_tokens:
                    processed_tokens.append(token)
                else:
                    processed_tokens.append('[VAR]')
            else:
                processed_tokens.append('[UNK]')
        
        return processed_tokens
    
    def _preprocess_formula(self, formula: str) -> str:
        """Preprocess formula for better tokenization"""
        # Convert to lowercase for consistency
        formula = formula.lower()
        
        # Replace common notation
        replacements = {
            '×': '*',
            '÷': '/',
            '²': '^2',
            '³': '^3',
            'π': 'pi',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'σ': 'sigma',
            'φ': 'phi',
            'ψ': 'psi',
            'ω': 'omega',
        }
        
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        
        # Add spaces around operators for better tokenization
        formula = re.sub(r'([+\-*/=<>!])', r' \1 ', formula)
        formula = re.sub(r'([(){}[\]])', r' \1 ', formula)
        
        # Remove extra whitespace
        formula = re.sub(r'\s+', ' ', formula).strip()
        
        return formula
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _is_variable(self, token: str) -> bool:
        """Check if token is a variable"""
        return token.isalpha() or (token[0].isalpha() and token[1:].isalnum())
    
    def encode(self, formula: str) -> Dict[str, Union[List[int], int]]:
        """Encode formula to token IDs"""
        tokens = self.tokenize(formula)
        
        # Add special tokens
        tokens = ['[BOS]'] + tokens + ['[EOS]']
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.token_to_id['[PAD]']] * (self.max_length - len(token_ids)))
        
        # Create attention mask
        attention_mask = [1 if tid != self.token_to_id['[PAD]'] else 0 for tid in token_ids]
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'tokens': tokens[:self.max_length] if len(tokens) > self.max_length else tokens
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to formula string"""
        tokens = []
        for tid in token_ids:
            if tid == self.token_to_id['[PAD]']:
                break  # Stop at padding
            elif tid == self.token_to_id['[BOS]'] or tid == self.token_to_id['[EOS]']:
                continue  # Skip special tokens
            else:
                token = self.id_to_token.get(tid, '[UNK]')
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def build_vocab_from_formulas(self, formulas: List[str]) -> None:
        """Build vocabulary from a list of formulas"""
        logger.info(f"Building vocabulary from {len(formulas)} formulas...")
        
        all_tokens = []
        for formula in formulas:
            tokens = self.tokenize(formula)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Add most frequent tokens to vocabulary
        current_id = max(self.token_to_id.values()) + 1
        
        for token, count in token_counts.most_common():
            if token not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        logger.info(f"Vocabulary size: {len(self.token_to_id)}")
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data.get('id_to_token', {}).items()}
        
        # Rebuild id_to_token if not present
        if not self.id_to_token:
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        self.vocab_size = vocab_data.get('vocab_size', len(self.token_to_id))
        self.max_length = vocab_data.get('max_length', 256)
        
        logger.info(f"Vocabulary loaded from {filepath}")

class MathParser:
    """Parser for mathematical expressions using SymPy"""
    
    def __init__(self):
        self.symbol_cache = {}
        
    def parse_expression(self, formula: str) -> Dict:
        """Parse mathematical expression and extract information"""
        try:
            # Preprocess formula
            processed = self._preprocess_for_sympy(formula)
            
            # Get symbols
            symbol_names = self._extract_symbols(processed)
            symbols_dict = self._create_symbols(symbol_names)
            
            # Parse expression
            expr = parse_expr(processed, transformations='all', local_dict=symbols_dict)
            
            return {
                'success': True,
                'expression': expr,
                'symbols': symbol_names,
                'latex': sp.latex(expr),
                'string': str(expr),
                'complexity': self._calculate_complexity(expr),
                'properties': self._analyze_properties(expr)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original': formula,
                'processed': processed if 'processed' in locals() else formula
            }
    
    def _preprocess_for_sympy(self, formula: str) -> str:
        """Preprocess formula for SymPy parsing"""
        # Handle common notation
        replacements = {
            '^': '**',  # Power notation
            'ln': 'log',  # Natural logarithm
            'arcsin': 'asin',
            'arccos': 'acos',
            'arctan': 'atan',
        }
        
        processed = formula
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed
    
    def _extract_symbols(self, formula: str) -> List[str]:
        """Extract variable symbols from formula"""
        # Find all potential variables
        variables = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', formula)
        
        # Filter out function names
        functions = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt',
            'abs', 'floor', 'ceil', 'max', 'min'
        }
        
        symbols = [var for var in variables if var not in functions]
        return list(set(symbols))
    
    def _create_symbols(self, symbol_names: List[str]) -> Dict:
        """Create SymPy symbol objects"""
        symbols_dict = {}
        
        for name in symbol_names:
            if name not in self.symbol_cache:
                self.symbol_cache[name] = symbols(name, real=True)
            symbols_dict[name] = self.symbol_cache[name]
        
        # Add constants
        symbols_dict.update({
            'pi': pi,
            'e': E,
            'PI': pi,
            'E': E
        })
        
        return symbols_dict
    
    def _calculate_complexity(self, expr) -> Dict:
        """Calculate expression complexity metrics"""
        try:
            return {
                'depth': self._expression_depth(expr),
                'nodes': len(expr.atoms()),
                'operations': len(expr.args),
                'functions': len([atom for atom in expr.atoms() if atom.is_Function]),
                'variables': len(expr.free_symbols)
            }
        except:
            return {'error': 'Could not calculate complexity'}
    
    def _expression_depth(self, expr, depth=0) -> int:
        """Calculate depth of expression tree"""
        if not hasattr(expr, 'args') or not expr.args:
            return depth
        return max(self._expression_depth(arg, depth + 1) for arg in expr.args)
    
    def _analyze_properties(self, expr) -> Dict:
        """Analyze mathematical properties of expression"""
        properties = {}
        
        try:
            # Check if expression has certain properties
            properties['is_polynomial'] = expr.is_polynomial()
            properties['is_rational'] = expr.is_rational
            properties['is_real'] = expr.is_real
            properties['is_complex'] = expr.is_complex
            
            # Count different types of functions
            atoms = expr.atoms()
            properties['has_trig'] = any(str(atom) in ['sin', 'cos', 'tan'] for atom in atoms)
            properties['has_exp'] = any(str(atom) == 'exp' for atom in atoms)
            properties['has_log'] = any(str(atom) == 'log' for atom in atoms)
            
            # Get free symbols
            properties['variables'] = [str(sym) for sym in expr.free_symbols]
            
        except Exception as e:
            properties['analysis_error'] = str(e)
        
        return properties

def create_tokenizer_from_dataset(metadata_file: str, 
                                 vocab_size: int = 10000,
                                 save_path: Optional[str] = None) -> FormulaTokenizer:
    """Create and train a tokenizer from a dataset"""
    
    # Load formulas from metadata
    formulas = []
    
    try:
        with open(metadata_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'formula' in entry:
                    formulas.append(entry['formula'])
    except Exception as e:
        logger.error(f"Failed to load formulas from {metadata_file}: {e}")
        return FormulaTokenizer(vocab_size=vocab_size)
    
    # Create and train tokenizer
    tokenizer = FormulaTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab_from_formulas(formulas)
    
    # Save if path provided
    if save_path:
        tokenizer.save_vocab(save_path)
    
    return tokenizer

# Example usage and testing
if __name__ == "__main__":
    # Test tokenizer
    tokenizer = FormulaTokenizer()
    
    test_formulas = [
        "sin(x) + cos(y)",
        "x^2 + y^2 = 1",
        "r = a * sin(n * theta)",
        "e^(i*pi) + 1 = 0"
    ]
    
    print("Testing Formula Tokenizer:")
    for formula in test_formulas:
        tokens = tokenizer.tokenize(formula)
        encoded = tokenizer.encode(formula)
        decoded = tokenizer.decode(encoded['input_ids'])
        
        print(f"\nFormula: {formula}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded['input_ids'][:10]}...")  # Show first 10 IDs
        print(f"Decoded: {decoded}")
    
    # Test parser
    parser = MathParser()
    
    print("\n\nTesting Math Parser:")
    for formula in test_formulas:
        result = parser.parse_expression(formula)
        
        print(f"\nFormula: {formula}")
        if result['success']:
            print(f"LaTeX: {result['latex']}")
            print(f"Complexity: {result['complexity']}")
            print(f"Properties: {result['properties']}")
        else:
            print(f"Error: {result['error']}")
    
    print("\nParser and tokenizer tests completed!")
