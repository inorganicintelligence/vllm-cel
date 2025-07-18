"""Main CLI interface for the analytical model."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .utils.csv_parser import analyze_benchmark_csv, BenchmarkCSVParser
from .hardware.registry import list_available_hardware, HardwareRegistry
from .models.registry import list_available_models, ModelRegistry


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analytical Model for MFU/MBU Calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Mixtral 8x7B benchmark results on A100
  python -m benchmarks.analytical_model \\
    --input logs/benchmark_results.csv \\
    --output logs/analysis_results.csv \\
    --hardware A100 \\
    --model Mixtral-8x7B
  
  # Analyze DeepSeek V3 with FP8 on H100
  python -m benchmarks.analytical_model \\
    --input logs/deepseek_results.csv \\
    --output logs/deepseek_analysis.csv \\
    --hardware H100 \\
    --model DeepSeek-V3 \\
    --dtype fp8
  
  # List available hardware and models
  python -m benchmarks.analytical_model --list-hardware
  python -m benchmarks.analytical_model --list-models
        """)
    
    # Main arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input CSV file with benchmark results'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        help='Output CSV file for analysis results'
    )
    
    parser.add_argument(
        '--hardware',
        type=str,
        choices=list_available_hardware(),
        help='Hardware specification (GPU type)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list_available_models(),
        help='Model specification'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='fp16',
        choices=['fp32', 'fp16', 'bf16', 'fp8', 'tf32'],
        help='Data type for computation (default: fp16)'
    )
    
    # Information arguments
    parser.add_argument(
        '--list-hardware',
        action='store_true',
        help='List available hardware specifications'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true', 
        help='List available model specifications'
    )
    
    parser.add_argument(
        '--validate-input',
        action='store_true',
        help='Validate input CSV format without running analysis'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Check for information-only commands
    if args.list_hardware or args.list_models:
        return True
    
    # For validation-only commands, only require input
    if args.validate_input:
        if not args.input:
            print("Error: --input is required for validation")
            return False
        return True
    
    # For analysis commands, require all main arguments
    if not args.input:
        print("Error: --input is required")
        return False
    
    if not args.output:
        print("Error: --output is required")
        return False
    
    if not args.hardware:
        print("Error: --hardware is required")
        return False
    
    if not args.model:
        print("Error: --model is required")
        return False
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        return False
    
    # Check that hardware supports the specified dtype
    try:
        hardware = HardwareRegistry.get_hardware(args.hardware)
        if args.dtype not in hardware.get_supported_dtypes():
            print(f"Error: {args.hardware} does not support dtype '{args.dtype}'")
            print(f"Supported dtypes: {hardware.get_supported_dtypes()}")
            return False
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    return True


def print_hardware_info():
    """Print information about available hardware."""
    print("Available Hardware Specifications:")
    print("=" * 40)
    
    for hw_name in list_available_hardware():
        try:
            hw = HardwareRegistry.get_hardware(hw_name)
            print(f"\n{hw.name}:")
            print(f"  Memory: {hw.memory_size_gb} GB")
            print(f"  Memory BW: {hw.peak_memory_bandwidth_gb_s} GB/s")
            print(f"  Tensor Cores: {hw.tensor_core_version}")
            print(f"  Supported dtypes: {', '.join(hw.get_supported_dtypes())}")
            
            # Show peak FLOPS for supported dtypes
            print("  Peak FLOPS:")
            for dtype in hw.get_supported_dtypes():
                flops = hw.get_peak_flops(dtype)
                print(f"    {dtype}: {flops} TFLOPS")
                
        except Exception as e:
            print(f"  Error loading {hw_name}: {e}")


def print_model_info():
    """Print information about available models."""
    print("Available Model Specifications:")
    print("=" * 40)
    
    for model_name in list_available_models():
        try:
            model = ModelRegistry.get_model(model_name)
            params = model.get_model_params()
            print(f"\n{model.name}:")
            print(f"  Hidden size: {params.get('hidden_size')}")
            print(f"  Layers: {params.get('num_layers')}")
            print(f"  Attention heads: {params.get('num_attention_heads')}")
            
            if 'num_experts' in params:
                print(f"  Experts: {params['num_experts']}")
                print(f"  Experts per token: {params['experts_per_token']}")
            
            if 'q_lora_rank' in params:
                print(f"  Q LoRA rank: {params['q_lora_rank']}")
                print(f"  KV LoRA rank: {params['kv_lora_rank']}")
                
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle information commands
    if args.list_hardware:
        print_hardware_info()
        return 0
    
    if args.list_models:
        print_model_info()
        return 0
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Validate input CSV format if requested
    if args.validate_input:
        if BenchmarkCSVParser.validate_csv_format(args.input):
            print(f"✓ Input CSV format is valid: {args.input}")
        else:
            print(f"✗ Input CSV format is invalid: {args.input}")
            return 1
        
        if not args.output:  # If only validating, we can exit here
            return 0
    
    # Run analysis
    try:
        if args.verbose:
            print(f"Starting analysis...")
            print(f"  Input: {args.input}")
            print(f"  Output: {args.output}")
            print(f"  Hardware: {args.hardware}")
            print(f"  Model: {args.model}")
            print(f"  Data type: {args.dtype}")
        
        analyze_benchmark_csv(
            input_csv_path=args.input,
            output_csv_path=args.output,
            hardware_name=args.hardware,
            model_name=args.model,
            dtype=args.dtype
        )
        
        if args.verbose:
            print("✓ Analysis completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())