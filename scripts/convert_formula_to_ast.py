#!/usr/bin/env python3
"""
Convert mathematical formulas to Abstract Syntax Tree (AST) representation.
This helps in understanding and parsing mathematical expressions for model training.
"""

import ast
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, sin, cos, tan, exp, log, sqrt, pi, E
import json
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FormulaToASTConverter:
    def __init__(self):
        # Common mathematical symbols
        self.common_symbols = ['x', 'y', 'z', 't', 'r', 'theta', 'phi', 'u', 'v', 'n', 'a', 'b', 'c']
        
        # Function mappings for different notation styles
        self.function_mappings = {
            'arcsin': 'asin',
            'arccos': 'acos', 
            'arctan': 'atan',
            'ln': 'log',
            'lg': 'log10',
        }
        
    def preprocess_formula(self, formula):
        """Preprocess formula to make it parseable"""
        if not formula or not isinstance(formula, str):
            return ""
        
        # Remove whitespace
        formula = formula.strip()
        
        # Handle implicit multiplication (e.g., "2x" -> "2*x")
        formula = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', formula)
        formula = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', formula)
        formula = re.sub(r'([a-zA-Z])\(', r'\1*(', formula)
        formula = re.sub(r'\)([a-zA-Z])', r')*\1', formula)
        
        # Handle common mathematical notation
        formula = formula.replace('^', '**')  # Power notation
        formula = formula.replace('×', '*')   # Multiplication symbol
        formula = formula.replace('÷', '/')   # Division symbol
        
        # Handle function name variations
        for old_name, new_name in self.function_mappings.items():
            formula = formula.replace(old_name, new_name)
        
        return formula
    
    def extract_symbols_from_formula(self, formula):
        """Extract mathematical symbols used in the formula"""
        # Find all variable-like patterns
        variables = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', formula)
        
        # Filter out function names
        function_names = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                         'exp', 'log', 'log10', 'sqrt', 'abs', 'floor', 'ceil', 'round',
                         'max', 'min', 'sum', 'prod']
        
        symbols_found = [var for var in variables if var not in function_names]
        return list(set(symbols_found))
    
    def parse_with_sympy(self, formula):
        """Parse formula using SymPy"""
        try:
            # Preprocess the formula
            processed = self.preprocess_formula(formula)
            
            # Extract symbols
            symbol_names = self.extract_symbols_from_formula(processed)
            
            # Create symbol objects
            symbol_dict = {}
            for name in symbol_names:
                symbol_dict[name] = symbols(name, real=True)
            
            # Add common constants
            symbol_dict.update({
                'pi': pi,
                'e': E,
                'PI': pi,
                'E': E
            })
            
            # Parse the expression
            expr = parse_expr(processed, transformations='all', local_dict=symbol_dict)
            
            return {
                'success': True,
                'expression': expr,
                'symbols': symbol_names,
                'string_repr': str(expr),
                'latex_repr': sp.latex(expr),
                'complexity': self.calculate_complexity(expr)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_formula': formula,
                'processed_formula': processed
            }
    
    def calculate_complexity(self, expr):
        """Calculate complexity metrics for the expression"""
        try:
            return {
                'atom_count': len(expr.atoms()),
                'operation_count': len(expr.args),
                'depth': self.expression_depth(expr),
                'function_count': len([atom for atom in expr.atoms() if atom.is_Function])
            }
        except:
            return {'error': 'Could not calculate complexity'}
    
    def expression_depth(self, expr, current_depth=0):
        """Calculate the depth of nested expressions"""
        if not hasattr(expr, 'args') or not expr.args:
            return current_depth
        
        return max(self.expression_depth(arg, current_depth + 1) for arg in expr.args)
    
    def extract_ast_structure(self, expr):
        """Extract AST-like structure from SymPy expression"""
        try:
            if expr.is_Atom:
                return {
                    'type': 'atom',
                    'value': str(expr),
                    'is_symbol': expr.is_Symbol,
                    'is_number': expr.is_Number
                }
            
            return {
                'type': 'expression',
                'func': str(type(expr).__name__),
                'args': [self.extract_ast_structure(arg) for arg in expr.args],
                'arg_count': len(expr.args)
            }
            
        except Exception as e:
            return {'type': 'error', 'message': str(e)}
    
    def analyze_formula_type(self, formula, ast_result):
        """Analyze the type of mathematical formula"""
        if not ast_result.get('success'):
            return {'type': 'unknown', 'reason': 'parse_failed'}
        
        formula_lower = formula.lower()
        expr = ast_result['expression']
        symbols = ast_result['symbols']
        
        # Determine formula type based on content and structure
        formula_types = []
        
        # Check for specific patterns
        if any(func in formula_lower for func in ['sin', 'cos', 'tan']):
            formula_types.append('trigonometric')
        
        if 'theta' in symbols or 'r' in symbols:
            formula_types.append('polar')
        
        if 't' in symbols and any(var in symbols for var in ['x', 'y']):
            formula_types.append('parametric')
        
        if len(symbols) >= 2 and any(var in symbols for var in ['x', 'y', 'z']):
            formula_types.append('multivariable')
        
        if any(func in formula_lower for func in ['exp', 'log']):
            formula_types.append('exponential_logarithmic')
        
        if '**' in formula or '^' in formula:
            formula_types.append('polynomial')
        
        if not formula_types:
            formula_types.append('algebraic')
        
        return {
            'types': formula_types,
            'primary_type': formula_types[0] if formula_types else 'unknown',
            'symbols_used': symbols,
            'symbol_count': len(symbols)
        }
    
    def convert_formula(self, formula):
        """Convert a single formula to AST representation"""
        # Parse with SymPy
        ast_result = self.parse_with_sympy(formula)
        
        # Analyze formula type
        type_analysis = self.analyze_formula_type(formula, ast_result)
        
        # Extract AST structure if parsing succeeded
        ast_structure = None
        if ast_result.get('success'):
            ast_structure = self.extract_ast_structure(ast_result['expression'])
        
        return {
            'original_formula': formula,
            'processed_formula': self.preprocess_formula(formula),
            'parse_result': ast_result,
            'type_analysis': type_analysis,
            'ast_structure': ast_structure,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
        }
    
    def convert_dataset_formulas(self, metadata_file, output_file=None):
        """Convert all formulas in a dataset to AST representation"""
        import pandas as pd
        
        # Load metadata
        formulas_data = []
        
        if Path(metadata_file).suffix == '.jsonl':
            with open(metadata_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    formulas_data.append(entry)
        else:
            with open(metadata_file, 'r') as f:
                formulas_data = json.load(f)
        
        logger.info(f"Converting {len(formulas_data)} formulas to AST...")
        
        # Convert each formula
        converted_data = []
        for i, entry in enumerate(formulas_data):
            formula = entry.get('formula', '')
            if formula:
                conversion_result = self.convert_formula(formula)
                
                # Add metadata context
                conversion_result['metadata'] = {
                    'image_path': entry.get('image_path'),
                    'description': entry.get('description'),
                    'tags': entry.get('tags', []),
                    'entry_index': i
                }
                
                converted_data.append(conversion_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(formulas_data)} formulas")
        
        # Save results
        if output_file is None:
            output_file = Path(metadata_file).parent / 'formulas_ast.json'
        
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2, default=str)
        
        logger.info(f"Saved AST conversions to {output_file}")
        
        # Generate summary
        self.generate_conversion_summary(converted_data)
        
        return converted_data
    
    def generate_conversion_summary(self, converted_data):
        """Generate summary of the conversion process"""
        total = len(converted_data)
        successful = sum(1 for item in converted_data if item['parse_result'].get('success'))
        failed = total - successful
        
        print(f"\n📊 FORMULA CONVERSION SUMMARY")
        print(f"="*40)
        print(f"Total formulas: {total}")
        print(f"Successfully parsed: {successful}")
        print(f"Parse failures: {failed}")
        print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "N/A")
        
        if successful > 0:
            # Analyze formula types
            all_types = []
            for item in converted_data:
                if item['parse_result'].get('success'):
                    types = item['type_analysis'].get('types', [])
                    all_types.extend(types)
            
            if all_types:
                from collections import Counter
                type_counts = Counter(all_types)
                print(f"\nFormula types found:")
                for formula_type, count in type_counts.most_common():
                    print(f"  {formula_type}: {count}")
        
        if failed > 0:
            print(f"\nSample parse failures:")
            failure_count = 0
            for item in converted_data:
                if not item['parse_result'].get('success') and failure_count < 3:
                    formula = item['original_formula']
                    error = item['parse_result'].get('error', 'Unknown error')
                    print(f"  '{formula}' -> {error}")
                    failure_count += 1

def main():
    """Main conversion process"""
    converter = FormulaToASTConverter()
    
    # Example: convert dataset metadata
    metadata_file = Path("dataset/metadata.jsonl")
    
    if metadata_file.exists():
        converter.convert_dataset_formulas(metadata_file)
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")
        
        # Demo with sample formulas
        sample_formulas = [
            "r = a * sin(n * theta)",
            "x = t * cos(t), y = t * sin(t)",
            "z = x^2 + y^2",
            "y = sin(x) + cos(2*x)",
            "r = 1 + cos(theta)"
        ]
        
        print("\nDemo: Converting sample formulas...")
        for formula in sample_formulas:
            result = converter.convert_formula(formula)
            print(f"\nFormula: {formula}")
            if result['parse_result'].get('success'):
                print(f"  Type: {result['type_analysis']['primary_type']}")
                print(f"  Symbols: {result['type_analysis']['symbols_used']}")
                print(f"  LaTeX: {result['parse_result']['latex_repr']}")
            else:
                print(f"  Error: {result['parse_result']['error']}")

if __name__ == "__main__":
    main()
