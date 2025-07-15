"""
Tokenizer for mathematical expressions and formulas.
Provides tokenization capabilities for converting mathematical text to tokens for model training.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, OrderedDict
import logging

logger = logging.getLogger(__name__)

class MathTokenizer:
    """Tokenizer specifically designed for mathematical expressions"""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize base vocabulary
        self.vocab = self._build_base_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Tokenization patterns
        self.patterns = self._create_patterns()
        
        logger.info(f"Initialized MathTokenizer with {len(self.vocab)} base tokens")
    
    def _build_base_vocab(self) -> List[str]:
        """Build base vocabulary for mathematical expressions"""
        vocab = []
        
        # Special tokens
        special_tokens = [
            '[PAD]', '[UNK]', '[BOS]', '[EOS]', '[MASK]',
            '[NUM]', '[VAR]', '[FUNC]'
        ]
        vocab.extend(special_tokens)
        
        # Mathematical operators
        operators = [
            '+', '-', '*', '/', '^', '**', '=', '==', '!=', 
            '<', '>', '<=', '>=', '~', '&', '|', '!',
            '∑', '∏', '∫', '∂', '∇', '∆'
        ]
        vocab.extend(operators)
        
        # Parentheses and brackets
        brackets = ['(', ')', '[', ']', '{', '}', '⟨', '⟩']
        vocab.extend(brackets)
        
        # Mathematical functions
        functions = [
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'coth',
            'exp', 'log', 'ln', 'log10', 'log2',
            'sqrt', 'cbrt', 'abs', 'sign',
            'floor', 'ceil', 'round', 'trunc',
            'max', 'min', 'gcd', 'lcm',
            'factorial', 'gamma', 'beta',
            'sum', 'prod', 'integral', 'diff',
            'limit', 'series'
        ]
        vocab.extend(functions)
        
        # Mathematical constants
        constants = [
            'pi', 'e', 'i', 'j', 'inf', 'nan',
            'euler', 'golden', 'catalan',
            'π', 'ℯ', '∞', 'ℵ'
        ]
        vocab.extend(constants)
        
        # Common variables (single letters)
        variables = [
            'x', 'y', 'z', 't', 'u', 'v', 'w',
            'r', 'θ', 'φ', 'ψ', 'ω', 'α', 'β', 'γ', 'δ', 'ε',
            'λ', 'μ', 'σ', 'τ', 'ρ', 'κ', 'η', 'ζ',
            'a', 'b', 'c', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 's',
            'A', 'B', 'C', 'D', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'S',
            'X', 'Y', 'Z', 'T', 'U', 'V', 'W', 'R'
        ]
        vocab.extend(variables)
        
        # Greek letters (commonly used in math)
        greek_letters = [
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
            'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
            'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
        ]
        vocab.extend(greek_letters)
        
        # Common mathematical notation
        notation = [
            'prime', 'double_prime', 'dot', 'ddot',
            'bar', 'hat', 'tilde', 'vec',
            'subscript', 'superscript',
            'frac', 'sqrt', 'root',
            'matrix', 'vector', 'tensor'
        ]
        vocab.extend(notation)
        
        # Punctuation and separators
        punctuation = [',', '.', ';', ':', '|', '||', '_', '^', "'", '"']
        vocab.extend(punctuation)
        
        # Common numbers (0-20)
        numbers = [str(i) for i in range(21)]
        vocab.extend(numbers)
        
        # Common fractions
        fractions = ['1/2', '1/3', '2/3', '1/4', '3/4', '1/5', '2/5', '3/5', '4/5']
        vocab.extend(fractions)
        
        return vocab
    
    def _create_patterns(self) -> List[Tuple[str, str]]:
        """Create regex patterns for tokenization"""
        patterns = [
            # Scientific notation
            (r'\d+\.?\d*[eE][+-]?\d+', 'NUMBER'),
            
            # Decimal numbers
            (r'\d+\.\d+', 'NUMBER'),
            
            # Integers
            (r'\d+', 'NUMBER'),
            
            # Fractions (simple format)
            (r'\d+/\d+', 'FRACTION'),
            
            # Mathematical functions (must come before variables)
            (r'\b(?:' + '|'.join([
                'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                'asin', 'acos', 'atan', 'atan2',
                'sinh', 'cosh', 'tanh', 'coth',
                'exp', 'log', 'ln', 'log10', 'log2',
                'sqrt', 'cbrt', 'abs', 'sign',
                'floor', 'ceil', 'round', 'trunc',
                'max', 'min', 'gcd', 'lcm'
            ]) + r')\b', 'FUNCTION'),
            
            # Mathematical constants
            (r'\b(?:pi|e|inf|nan|euler|golden|catalan)\b', 'CONSTANT'),
            
            # Greek letters (full names)
            (r'\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b', 'GREEK'),
            
            # Variables (letters, possibly with subscripts/superscripts)
            (r'[a-zA-Z][a-zA-Z0-9_]*', 'VARIABLE'),
            
            # Two-character operators
            (r'\*\*|==|!=|<=|>=|//', 'OPERATOR'),
            
            # Single-character operators
            (r'[+\-*/^=<>~&|!]', 'OPERATOR'),
            
            # Brackets and parentheses
            (r'[()[\]{}⟨⟩]', 'BRACKET'),
            
            # Unicode mathematical symbols
            (r'[∑∏∫∂∇∆π℮∞ℵ]', 'SYMBOL'),
            
            # Punctuation
            (r'[,.;:|\'"]', 'PUNCTUATION'),
            
            # Whitespace (to be filtered out)
            (r'\s+', 'WHITESPACE'),
            
            # Any other character
            (r'.', 'OTHER')
        ]
        
        return patterns
    
    def tokenize(self, text: str, normalize: bool = True) -> List[str]:
        """Tokenize mathematical text into tokens"""
        if normalize:
            text = self._normalize_text(text)
        
        tokens = []
        pos = 0
        
        while pos < len(text):
            matched = False
            
            for pattern, token_type in self.patterns:
                regex = re.compile(pattern)
                match = regex.match(text, pos)
                
                if match:
                    token = match.group(0)
                    
                    # Skip whitespace
                    if token_type == 'WHITESPACE':
                        pos = match.end()
                        matched = True
                        break
                    
                    # Process token based on type
                    if token_type == 'NUMBER':
                        tokens.append('[NUM]')
                    elif token_type == 'VARIABLE' and len(token) == 1:
                        tokens.append(token)  # Single letter variables
                    elif token_type == 'VARIABLE':
                        tokens.append('[VAR]')  # Multi-letter variables
                    elif token_type == 'FUNCTION':
                        tokens.append(token)
                    elif token in self.token_to_id:
                        tokens.append(token)
                    else:
                        tokens.append('[UNK]')
                    
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                # Skip unknown character
                pos += 1
        
        return tokens
    
    def _normalize_text(self, text: str) -> str:
        """Normalize mathematical text"""
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Replace common mathematical notation
        replacements = {
            '×': '*',
            '÷': '/',
            '²': '^2',
            '³': '^3',
            '√': 'sqrt',
            '∞': 'inf',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'ζ': 'zeta',
            'η': 'eta',
            'θ': 'theta',
            'ι': 'iota',
            'κ': 'kappa',
            'λ': 'lambda',
            'μ': 'mu',
            'ν': 'nu',
            'ξ': 'xi',
            'ο': 'omicron',
            'π': 'pi',
            'ρ': 'rho',
            'σ': 'sigma',
            'τ': 'tau',
            'υ': 'upsilon',
            'φ': 'phi',
            'χ': 'chi',
            'ψ': 'psi',
            'ω': 'omega'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, Union[List[int], List[str]]]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['[BOS]'] + tokens + ['[EOS]']
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            tokens = tokens[:self.max_length]
        else:
            pad_length = self.max_length - len(token_ids)
            token_ids.extend([self.token_to_id['[PAD]']] * pad_length)
            tokens.extend(['[PAD]'] * pad_length)
        
        # Create attention mask
        attention_mask = [1 if tid != self.token_to_id['[PAD]'] else 0 for tid in token_ids]
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                
                if skip_special_tokens and token in ['[BOS]', '[EOS]', '[PAD]']:
                    continue
                
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def add_tokens(self, new_tokens: List[str]) -> int:
        """Add new tokens to vocabulary"""
        added_count = 0
        
        for token in new_tokens:
            if token not in self.token_to_id and len(self.vocab) < self.vocab_size:
                token_id = len(self.vocab)
                self.vocab.append(token)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                added_count += 1
        
        return added_count
    
    def build_vocab_from_corpus(self, texts: List[str], min_frequency: int = 2) -> None:
        """Build vocabulary from a corpus of mathematical texts"""
        logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Tokenize all texts and count frequencies
        token_counter = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)
        
        # Add frequent tokens to vocabulary
        frequent_tokens = [token for token, count in token_counter.items() 
                          if count >= min_frequency and token not in self.token_to_id]
        
        added = self.add_tokens(frequent_tokens)
        logger.info(f"Added {added} new tokens to vocabulary")
        
        # Log vocabulary statistics
        logger.info(f"Final vocabulary size: {len(self.vocab)}")
        logger.info(f"Most common tokens: {token_counter.most_common(10)}")
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data.get('id_to_token', {}).items()}
        
        # Rebuild id_to_token if not present
        if not self.id_to_token:
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        self.vocab_size = vocab_data.get('vocab_size', len(self.vocab))
        self.max_length = vocab_data.get('max_length', 256)
        
        logger.info(f"Vocabulary loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.vocab)
    
    def get_token_stats(self, text: str) -> Dict:
        """Get tokenization statistics for a text"""
        tokens = self.tokenize(text)
        
        stats = {
            'num_tokens': len(tokens),
            'num_unique_tokens': len(set(tokens)),
            'unknown_tokens': sum(1 for token in tokens if token == '[UNK]'),
            'number_tokens': sum(1 for token in tokens if token == '[NUM]'),
            'variable_tokens': sum(1 for token in tokens if token == '[VAR]'),
            'token_distribution': Counter(tokens)
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = MathTokenizer(vocab_size=5000, max_length=128)
    
    test_expressions = [
        "sin(x) + cos(y) = 1",
        "x^2 + y^2 ≤ r^2",
        "∫ f(x) dx from 0 to π",
        "e^(iπ) + 1 = 0",
        "∑(n=1 to ∞) 1/n^2 = π²/6",
        "f'(x) = lim(h→0) [f(x+h) - f(x)]/h"
    ]
    
    print("Testing Math Tokenizer:")
    print(f"Base vocabulary size: {tokenizer.get_vocab_size()}")
    
    for expr in test_expressions:
        print(f"\nExpression: {expr}")
        
        # Tokenize
        tokens = tokenizer.tokenize(expr)
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # Encode
        encoded = tokenizer.encode(expr)
        print(f"Encoded length: {len(encoded['input_ids'])}")
        print(f"Attention mask sum: {sum(encoded['attention_mask'])}")
        
        # Decode
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"Decoded: {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
        
        # Stats
        stats = tokenizer.get_token_stats(expr)
        print(f"Stats: {stats['num_tokens']} tokens, {stats['unknown_tokens']} unknown")
    
    print("\nTokenizer tests completed!")
