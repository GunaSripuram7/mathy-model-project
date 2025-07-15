# Generated Outputs

This directory contains generated mathematical art images from test prompts and model inference.

## Contents

- `prompt_test_*.png` - Images generated from test prompts during validation
- `comparison_grids/` - Side-by-side comparisons of original vs generated images  
- `sample_generations/` - Random samples generated during training
- `evaluation_outputs/` - Images used for quantitative evaluation

## File Naming Convention

- `prompt_test_{n}.png` - Test prompt number n
- `epoch_{e}_sample_{s}.png` - Sample s generated at epoch e
- `comparison_epoch_{e}.png` - Comparison grid at epoch e
- `eval_batch_{b}.png` - Evaluation batch b

## Example Test Prompts

1. "r = sin(3*theta)" - Rose curve with 3 petals
2. "x = t*cos(t), y = t*sin(t)" - Archimedean spiral  
3. "z = x^2 + y^2" - Paraboloid surface
4. "spiral with golden ratio" - Fibonacci spiral
5. "mandelbrot fractal" - Mandelbrot set visualization

Generated images will appear here during model training and inference.
