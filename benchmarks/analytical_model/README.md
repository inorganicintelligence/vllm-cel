# Analytical Model for MFU/MBU Calculation

A comprehensive analytical framework for calculating Model FLOPs Utilization (MFU) and Memory Bandwidth Utilization (MBU) from vLLM benchmark results. This tool provides detailed performance analysis across different hardware platforms and model architectures, with support for various parallelism strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Architecture](#architecture)
- [Supported Hardware](#supported-hardware)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Understanding Results](#understanding-results)
- [Extending the Framework](#extending-the-framework)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This analytical model bridges the gap between theoretical hardware capabilities and actual performance by:

1. **Calculating theoretical FLOPs and memory requirements** for specific model architectures
2. **Measuring actual throughput** from benchmark results
3. **Computing utilization percentages** (MFU/MBU) to identify bottlenecks
4. **Providing phase-specific analysis** for prefill vs decode operations

The framework is designed to help ML engineers and researchers optimize their inference deployments by understanding where performance bottlenecks occur.

## Features

### 🔧 **Core Capabilities**
- **Multi-Hardware Support**: A100, H100, H200, B200 with accurate specifications
- **Multi-Model Support**: Mixtral 8x7B, DeepSeek V3 with architectural details
- **Parallelism Modeling**: Tensor Parallelism (TP) and Expert Parallelism (EP)
- **Phase Analysis**: Separate calculations for prefill and decode phases
- **Data Type Support**: FP32, FP16, BF16, FP8, TF32 with hardware-specific constraints

### 📊 **Analysis Features**
- **MFU Calculation**: Model FLOPs Utilization as percentage of peak hardware capability
- **MBU Calculation**: Memory Bandwidth Utilization as percentage of peak bandwidth
- **Per-GPU Metrics**: Accounts for distributed workloads across multiple GPUs
- **Communication Overhead**: Models All2All communication costs for expert parallelism
- **Memory Optimization**: Accounts for Flash Attention v2 and compressed KV cache (MLA)

### 🛠 **Integration Features**
- **CSV Pipeline**: Direct integration with vLLM benchmark CSV output
- **CLI Interface**: Command-line tool with validation and verbose modes
- **Extensible Design**: Easy addition of new hardware specs and model architectures
- **Comprehensive Testing**: Unit tests and integration validation

## Installation

No additional installation required - the analytical model uses only Python standard library and existing vLLM dependencies.

```bash
# Verify installation by listing available hardware
python -m benchmarks.analytical_model --list-hardware
```

## File Structure

```
analytical_model/
├── __init__.py                 # Package initialization and exports
├── __main__.py                 # Module entry point for CLI usage
├── main.py                     # Main CLI interface and argument parsing
├── README.md                   # This comprehensive documentation
│
├── hardware/                   # Hardware specifications and registry
│   ├── __init__.py            # Hardware package exports
│   ├── base.py                # BaseHardware abstract class
│   ├── nvidia_gpus.py         # NVIDIA GPU implementations (A100, H100, H200, B200)
│   └── registry.py            # Hardware registry and factory pattern
│
├── models/                     # Model specifications and implementations
│   ├── __init__.py            # Models package exports
│   ├── base.py                # BaseModel abstract class and common utilities
│   ├── mixtral.py             # Mixtral 8x7B implementation with MoE modeling
│   ├── deepseek.py            # DeepSeek V3 implementation with MLA support
│   └── registry.py            # Model registry and factory pattern
│
├── calculators/                # Core calculation engines
│   ├── __init__.py            # Calculators package exports
│   └── utilization.py         # MFU/MBU calculation logic and metrics
│
├── utils/                      # Utility functions and constants
│   ├── __init__.py            # Utils package exports
│   ├── csv_parser.py          # CSV parsing and analysis output generation
│   └── constants.py           # Common constants and data type definitions
│
└── tests/                      # Test suite
    ├── __init__.py            # Test package initialization
    └── test_basic.py          # Basic functionality and integration tests
```

### Key Files Explained

#### **Hardware Layer (`hardware/`)**

- **`base.py`**: Defines `BaseHardware` abstract class with methods for peak FLOPS calculation, supported data types, and memory specifications
- **`nvidia_gpus.py`**: Implements specific GPU classes (`A100Hardware`, `H100Hardware`, etc.) with accurate specifications
- **`registry.py`**: Provides hardware registry pattern for easy hardware selection and instantiation

#### **Model Layer (`models/`)**

- **`base.py`**: Defines `BaseModel` abstract class with methods for FLOP and memory calculations, includes helper methods for attention and MoE computations
- **`mixtral.py`**: Implements Mixtral 8x7B with 8 experts, top-2 routing, and GQA (Grouped Query Attention)
- **`deepseek.py`**: Implements DeepSeek V3 with 256 experts, top-8 routing, and MLA (Multi-Latent Attention) compression
- **`registry.py`**: Provides model registry for easy model selection and instantiation

#### **Calculator Layer (`calculators/`)**

- **`utilization.py`**: Core calculation engine that combines hardware specs, model metrics, and benchmark results to compute MFU/MBU

#### **Utility Layer (`utils/`)**

- **`csv_parser.py`**: Handles parsing of benchmark CSV input and generation of analysis CSV output
- **`constants.py`**: Common constants for data types, unit conversions, and model architectures

## Quick Start

### 1. **Run Analysis on Existing Benchmark Results**

```bash
# Analyze Mixtral 8x7B results on A100
python -m benchmarks.analytical_model \
  --input logs/testbenchmark_results.csv \
  --output logs/analysis_results.csv \
  --hardware a100 \
  --model mixtral-8x7b \
  --dtype fp16 \
  --verbose
```

### 2. **Explore Available Options**

```bash
# List supported hardware
python -m benchmarks.analytical_model --list-hardware

# List supported models  
python -m benchmarks.analytical_model --list-models

# Validate CSV format
python -m benchmarks.analytical_model --validate-input --input your_results.csv
```

### 3. **Programmatic Usage**

```python
from benchmarks.analytical_model import (
    get_hardware, get_model, UtilizationCalculator, analyze_benchmark_csv
)

# Quick analysis
analyze_benchmark_csv(
    input_csv_path="benchmark_results.csv",
    output_csv_path="analysis_results.csv", 
    hardware_name="a100",
    model_name="mixtral-8x7b",
    dtype="fp16"
)

# Advanced usage
hardware = get_hardware("h100")
model = get_model("deepseek-v3")
calculator = UtilizationCalculator(hardware, model, dtype="fp8")
```

## Detailed Usage

### Command Line Interface

The analytical model provides a comprehensive CLI with the following options:

```bash
python -m benchmarks.analytical_model [OPTIONS]
```

#### **Required Arguments (for analysis)**
- `--input, -i`: Input CSV file with benchmark results
- `--output, -o`: Output CSV file for analysis results  
- `--hardware`: Hardware specification (`a100`, `h100`, `h200`, `b200`)
- `--model`: Model specification (`mixtral-8x7b`, `deepseek-v3`)

#### **Optional Arguments**
- `--dtype`: Data type for computation (`fp16`, `fp8`, `bf16`, `tf32`, `fp32`) [default: fp16]
- `--verbose, -v`: Enable verbose output with detailed progress information
- `--validate-input`: Validate input CSV format without running analysis
- `--list-hardware`: Display all available hardware specifications
- `--list-models`: Display all available model specifications

### Input CSV Format

The tool expects CSV input from `run_mixtral_benchmark_sweep.sh` with the following required columns:

```csv
input_len,output_len,batch_size,tp_size,ep_enabled,system_throughput,request_throughput,avg_prefill_time,avg_decode_time,mean_ttft,mean_tpot,itl_mean,itl_median,itl_p99
512,512,8,2,true,1000.0,1.0,0.1,5.0,0.05,0.01,0.01,0.009,0.02
```

**Key Columns Explained:**
- `input_len`: Input sequence length (tokens)
- `output_len`: Output sequence length (tokens)  
- `batch_size`: Batch size used in benchmark
- `tp_size`: Tensor parallel size (number of GPUs)
- `ep_enabled`: Whether expert parallelism is enabled (true/false)
- `system_throughput`: Total tokens/second across all GPUs
- `avg_prefill_time`: Average prefill time in seconds
- `avg_decode_time`: Average decode time in seconds

### Output CSV Format

The analysis adds comprehensive metrics to the input CSV:

#### **Utilization Metrics** (as ratios, multiply by 100 for percentages)
- `total_mfu`: Overall Model FLOPs Utilization
- `total_mbu`: Overall Memory Bandwidth Utilization  
- `prefill_mfu`: Prefill phase Model FLOPs Utilization
- `prefill_mbu`: Prefill phase Memory Bandwidth Utilization
- `decode_mfu`: Decode phase Model FLOPs Utilization
- `decode_mbu`: Decode phase Memory Bandwidth Utilization

#### **Raw Performance Metrics**
- `actual_total_flops_tflops`: Actual FLOP throughput (TFLOPS) 
- `actual_prefill_flops_tflops`: Prefill FLOP throughput (TFLOPS)
- `actual_decode_flops_tflops`: Decode FLOP throughput (TFLOPS)
- `actual_total_memory_bw_gb_s`: Actual memory bandwidth (GB/s)
- `actual_prefill_memory_bw_gb_s`: Prefill memory bandwidth (GB/s)
- `actual_decode_memory_bw_gb_s`: Decode memory bandwidth (GB/s)

#### **Analysis Metadata**
- `hardware`: Hardware specification used for analysis
- `model`: Model specification used for analysis
- `dtype`: Data type used for computation

## Architecture

### Calculation Flow

```
Benchmark CSV Input
        ↓
Parse & Validate
        ↓
Hardware Specs ← → Model Architecture ← → Parallelism Config
        ↓
Theoretical Calculations
  • FLOPs per operation
  • Memory per operation  
  • Per-GPU adjustments
        ↓
Actual Performance Extraction
  • Tokens/sec from benchmarks
  • Time measurements
        ↓
Utilization Computation
  • MFU = Actual FLOPS / Peak FLOPS
  • MBU = Actual Memory BW / Peak Memory BW
        ↓
Analysis CSV Output
```

### Design Patterns

#### **Registry Pattern**
Both hardware and models use a registry pattern for easy extensibility:

```python
# Hardware registry
from benchmarks.analytical_model.hardware.registry import HardwareRegistry

@dataclass
class MyCustomGPU(BaseHardware):
    # Implementation
    pass

HardwareRegistry.register("my_gpu", MyCustomGPU)
hardware = HardwareRegistry.get_hardware("my_gpu")
```

#### **Factory Pattern**
Convenient factory functions abstract the registry complexity:

```python
from benchmarks.analytical_model import get_hardware, get_model

hardware = get_hardware("a100")  # Returns A100Hardware instance
model = get_model("mixtral-8x7b")  # Returns Mixtral8x7B instance
```

#### **Strategy Pattern**
Different calculation strategies for different model architectures:

```python
class BaseModel:
    def calculate_attention_flops(self, ...):
        # Standard attention calculation
        pass

class DeepSeekV3(BaseModel):
    def _calculate_mla_attention_flops(self, ...):
        # Multi-Latent Attention specific calculation
        pass
```

## Supported Hardware

### NVIDIA GPUs

| Hardware | Memory | Memory BW | Tensor Cores | FP16 FLOPS | FP8 FLOPS | Notes |
|----------|--------|-----------|--------------|------------|-----------|-------|
| **A100** | 80 GB | 2,039 GB/s | 3rd Gen | 312 TFLOPS | N/A | Established baseline |
| **H100** | 80 GB | 3,350 GB/s | 4th Gen | 989 TFLOPS | 1,979 TFLOPS | FP8 support |
| **H200** | 141 GB | 4,800 GB/s | 4th Gen | 989 TFLOPS | 1,979 TFLOPS | Enhanced memory |
| **B200** | 192 GB | 8,000 GB/s | 5th Gen | 2,250 TFLOPS | 4,500 TFLOPS | Next-gen architecture |

### Data Type Support

Each hardware platform supports different computational data types:

```python
# Check supported data types
hardware = get_hardware("h100")
print(hardware.get_supported_dtypes())  # ['fp8', 'fp16', 'bf16', 'tf32', 'fp32']

# Get peak FLOPS for specific dtype
peak_fp8 = hardware.get_peak_flops("fp8")  # 1979.0 TFLOPS
peak_fp16 = hardware.get_peak_flops("fp16")  # 989.0 TFLOPS
```

## Supported Models

### Mixtral 8x7B

**Architecture Details:**
- **Type**: Mixture of Experts (MoE)
- **Experts**: 8 total, top-2 routing
- **Hidden Size**: 4,096
- **Layers**: 32
- **Attention**: Grouped Query Attention (GQA) with 32 heads, 8 KV heads
- **Intermediate Size**: 14,336 per expert

**Parallelism Support:**
- **Tensor Parallelism**: Attention weights sharded across GPUs
- **Expert Parallelism**: Each GPU owns subset of complete experts
- **Mixed TP+EP**: Attention uses TP, experts use EP

**Memory Optimizations:**
- Flash Attention v2 for memory-efficient attention computation
- GQA reduces KV cache memory requirements

### DeepSeek V3

**Architecture Details:**
- **Type**: Large-scale Mixture of Experts with Multi-Latent Attention
- **Experts**: 256 total, top-8 routing
- **Hidden Size**: 7,168  
- **Layers**: 61
- **Attention**: Multi-Latent Attention (MLA) with compression
- **Intermediate Size**: 18,432 per expert

**MLA (Multi-Latent Attention) Features:**
- **Compressed Representations**: 
  - Q LoRA rank: 1,536 (vs 7,168 full)
  - KV LoRA rank: 512 (vs 7,168 full)
- **Memory Efficiency**: ~70% reduction in KV cache size
- **Computational Modes**: Compute-friendly (prefill) vs data-movement friendly (decode)

**Parallelism Support:**
- **Massive Expert Parallelism**: 256 experts distributed across many GPUs
- **Load Balancing**: Redundant experts for load balancing
- **Tensor Parallelism**: Applied to non-expert layers

## API Reference

### Core Classes

#### **BaseHardware**

```python
@dataclass
class BaseHardware(ABC):
    name: str
    peak_memory_bandwidth_gb_s: float
    memory_size_gb: float
    tensor_core_version: str
    
    @abstractmethod
    def get_peak_flops(self, dtype: str) -> float:
        """Get peak FLOPS for specified data type."""
        pass
    
    @abstractmethod  
    def get_supported_dtypes(self) -> list[str]:
        """Get list of supported data types."""
        pass
```

#### **BaseModel**

```python
class BaseModel(ABC):
    @abstractmethod
    def calculate_total_flops_per_gpu(
        self, seq_len: int, batch_size: int, 
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate total FLOPs per GPU."""
        pass
    
    @abstractmethod
    def calculate_total_memory_per_gpu(
        self, seq_len: int, batch_size: int,
        parallelism_config: ParallelismConfig  
    ) -> float:
        """Calculate total memory access per GPU."""
        pass
```

#### **UtilizationCalculator**

```python
class UtilizationCalculator:
    def __init__(self, hardware: BaseHardware, model: BaseModel, dtype: str):
        """Initialize calculator with hardware, model, and data type."""
        
    def calculate_utilization(
        self, benchmark_results: BenchmarkResults
    ) -> UtilizationMetrics:
        """Calculate MFU/MBU from benchmark results."""
```

### Configuration Classes

#### **ParallelismConfig**

```python
@dataclass
class ParallelismConfig:
    tp_size: int = 1                    # Tensor parallel size
    ep_enabled: bool = False            # Expert parallelism enabled
    dp_size: int = 1                    # Data parallel size  
    use_flash_attention: bool = True    # Flash Attention v2
```

#### **ModelMetrics**

```python
@dataclass
class ModelMetrics:
    total_flops: float           # Total FLOPs per GPU (TFLOPS)
    prefill_flops: float         # Prefill FLOPs per GPU (TFLOPS)
    decode_flops: float          # Decode FLOPs per GPU (TFLOPS)
    total_memory_gb: float       # Total memory access per GPU (GB)
    prefill_memory_gb: float     # Prefill memory access per GPU (GB)
    decode_memory_gb: float      # Decode memory access per GPU (GB)
```

## Examples

### Example 1: Basic Analysis

```python
from benchmarks.analytical_model import analyze_benchmark_csv

# Simple analysis with default settings
analyze_benchmark_csv(
    input_csv_path="benchmark_results.csv",
    output_csv_path="analysis_results.csv",
    hardware_name="a100", 
    model_name="mixtral-8x7b"
)
```

### Example 2: Advanced Programmatic Usage

```python
from benchmarks.analytical_model import (
    get_hardware, get_model, UtilizationCalculator, 
    BenchmarkResults, ParallelismConfig
)

# Load specifications
hardware = get_hardware("h100") 
model = get_model("deepseek-v3")

# Create calculator with FP8 precision
calculator = UtilizationCalculator(hardware, model, dtype="fp8")

# Create benchmark results (normally parsed from CSV)
benchmark = BenchmarkResults(
    input_len=1024,
    output_len=2048, 
    batch_size=16,
    tp_size=8,
    ep_enabled=True,
    system_throughput=2500.0,  # tokens/sec
    request_throughput=1.22,   # requests/sec
    avg_prefill_time=0.208,    # seconds
    avg_decode_time=33.67      # seconds
)

# Calculate utilization
utilization = calculator.calculate_utilization(benchmark)

print(f"Total MFU: {utilization.total_mfu:.3f} ({utilization.total_mfu*100:.1f}%)")
print(f"Total MBU: {utilization.total_mbu:.3f} ({utilization.total_mbu*100:.1f}%)")
print(f"Prefill MFU: {utilization.prefill_mfu:.3f} ({utilization.prefill_mfu*100:.1f}%)")
print(f"Decode MFU: {utilization.decode_mfu:.3f} ({utilization.decode_mfu*100:.1f}%)")
```

### Example 3: Batch Analysis with Filtering

```python
import pandas as pd
from benchmarks.analytical_model.utils.csv_parser import BenchmarkCSVParser
from benchmarks.analytical_model import get_hardware, get_model, UtilizationCalculator

# Parse benchmark results
results = BenchmarkCSVParser.parse_benchmark_csv("benchmark_results.csv")

# Setup calculator
hardware = get_hardware("a100")
model = get_model("mixtral-8x7b") 
calculator = UtilizationCalculator(hardware, model, "fp16")

# Analyze and filter
analysis_data = []
for result in results:
    if result.tp_size == 2 and result.ep_enabled:  # Filter specific configs
        utilization = calculator.calculate_utilization(result)
        analysis_data.append({
            'config': f"{result.input_len}x{result.output_len}_bs{result.batch_size}",
            'mfu': utilization.total_mfu,
            'mbu': utilization.total_mbu,
            'bottleneck': 'compute' if utilization.total_mfu > utilization.total_mbu else 'memory'
        })

# Convert to DataFrame for easy analysis
df = pd.DataFrame(analysis_data)
print(df)
```

### Example 4: Hardware Comparison

```python
from benchmarks.analytical_model import get_hardware, get_model, UtilizationCalculator

# Compare same workload across different hardware
hardware_types = ["a100", "h100", "h200"]
model = get_model("deepseek-v3")

# Sample benchmark result
benchmark = BenchmarkResults(
    input_len=512, output_len=1024, batch_size=8, tp_size=4, ep_enabled=True,
    system_throughput=1500.0, request_throughput=0.73,
    avg_prefill_time=0.068, avg_decode_time=20.5
)

print("Hardware Comparison:")
print("-" * 50)
for hw_name in hardware_types:
    hardware = get_hardware(hw_name)
    calculator = UtilizationCalculator(hardware, model, "fp16")
    utilization = calculator.calculate_utilization(benchmark)
    
    print(f"{hardware.name:>6}: MFU={utilization.total_mfu:.3f} MBU={utilization.total_mbu:.3f}")
```

### Example 5: Custom Hardware Extension

```python
from benchmarks.analytical_model.hardware.base import BaseHardware
from benchmarks.analytical_model.hardware.registry import HardwareRegistry

@dataclass  
class CustomGPU(BaseHardware):
    """Custom GPU specification."""
    
    def __init__(self):
        super().__init__(
            name="CustomGPU",
            peak_memory_bandwidth_gb_s=5000.0,
            memory_size_gb=128.0,
            tensor_core_version="Custom",
            base_clock_ghz=2.5,
            memory_clock_ghz=3.5,
            memory_bus_width_bits=6144
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        flops_map = {
            'fp16': 1500.0,
            'fp8': 3000.0,
            'fp32': 100.0,
        }
        return flops_map.get(dtype, 0.0)
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp16', 'fp8', 'fp32']

# Register and use custom hardware
HardwareRegistry.register("custom_gpu", CustomGPU)
hardware = get_hardware("custom_gpu")
print(f"Custom GPU FP8 Peak: {hardware.get_peak_flops('fp8')} TFLOPS")
```

## Understanding Results

### Interpreting MFU (Model FLOPs Utilization)

**MFU = (Actual FLOPs/sec) / (Peak Hardware FLOPs/sec)**

- **High MFU (>50%)**: Compute-bound workload, good hardware utilization
- **Medium MFU (20-50%)**: Balanced or mildly compute-bound
- **Low MFU (<20%)**: Memory-bound or inefficient computation

### Interpreting MBU (Memory Bandwidth Utilization)

**MBU = (Actual Memory Bandwidth) / (Peak Memory Bandwidth)**

- **High MBU (>80%)**: Memory-bound workload, bandwidth is the bottleneck
- **Medium MBU (40-80%)**: Balanced memory usage
- **Low MBU (<40%)**: Compute-bound or efficient memory usage

### Common Patterns

#### **Memory-Bound Workloads (MBU >> MFU)**
```
Example: MFU=15%, MBU=85%
Interpretation: Limited by memory bandwidth, not compute
Optimizations: Reduce memory transfers, increase arithmetic intensity
```

#### **Compute-Bound Workloads (MFU >> MBU)** 
```
Example: MFU=70%, MBU=30%
Interpretation: Limited by compute throughput
Optimizations: Improve kernel efficiency, reduce compute complexity
```

#### **Balanced Workloads (MFU ≈ MBU)**
```
Example: MFU=45%, MBU=50% 
Interpretation: Well-balanced utilization
Optimizations: General efficiency improvements
```

### Phase-Specific Analysis

#### **Prefill vs Decode Characteristics**

**Prefill (typically compute-bound):**
- Processes entire input sequence in parallel
- Higher arithmetic intensity (more FLOPs per byte)
- Usually shows higher MFU, lower MBU

**Decode (typically memory-bound):**
- Processes one token at a time
- Lower arithmetic intensity
- Heavy KV cache access
- Usually shows lower MFU, higher MBU

### Troubleshooting High MBU Values

If you see MBU > 100%, this typically indicates:

1. **Theoretical Model Limitations**: Our analytical model may overestimate memory requirements
2. **Cache Effects**: Real hardware has sophisticated caching that reduces actual bandwidth needs  
3. **Kernel Optimizations**: Optimized kernels may access memory more efficiently than theoretical calculations
4. **Measurement Granularity**: Benchmark measurements may not capture all memory access patterns

This is normal and provides insight into where theoretical models diverge from practice.

## Extending the Framework

### Adding New Hardware

1. **Create Hardware Class**:
```python
from benchmarks.analytical_model.hardware.base import BaseHardware

@dataclass
class NewGPU(BaseHardware):
    def __init__(self):
        super().__init__(
            name="NewGPU",
            peak_memory_bandwidth_gb_s=6000.0,
            memory_size_gb=200.0,
            tensor_core_version="NextGen",
            base_clock_ghz=3.0,
            memory_clock_ghz=4.0,
            memory_bus_width_bits=8192
        )
    
    def get_peak_flops(self, dtype: str) -> float:
        # Implementation specific to new GPU
        pass
    
    def get_supported_dtypes(self) -> list[str]:
        return ['fp4', 'fp8', 'fp16', 'bf16']
```

2. **Register Hardware**:
```python
from benchmarks.analytical_model.hardware.registry import HardwareRegistry
HardwareRegistry.register("new_gpu", NewGPU)
```

### Adding New Models

1. **Create Model Class**:
```python
from benchmarks.analytical_model.models.base import BaseModel, ParallelismConfig

class NewModel(BaseModel):
    def __init__(self):
        super().__init__("NewModel")
        # Define model parameters
        self.hidden_size = 8192
        self.num_layers = 48
        # ... other parameters
    
    def calculate_total_flops_per_gpu(self, seq_len, batch_size, parallelism_config):
        # Implement model-specific FLOP calculation
        pass
    
    def calculate_total_memory_per_gpu(self, seq_len, batch_size, parallelism_config):
        # Implement model-specific memory calculation  
        pass
    
    # Implement other required methods...
```

2. **Register Model**:
```python
from benchmarks.analytical_model.models.registry import ModelRegistry
ModelRegistry.register("new_model", NewModel)
```

### Adding New Calculation Methods

For specialized attention mechanisms or novel architectures:

```python
def calculate_custom_attention_flops(self, seq_len, batch_size, config):
    """Custom attention mechanism FLOPs calculation."""
    # Implement specialized computation
    pass

def calculate_custom_memory_pattern(self, seq_len, batch_size, config):
    """Custom memory access pattern calculation."""
    # Implement specialized memory modeling
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m benchmarks.analytical_model.tests.test_basic

# Run with verbose output
python -m benchmarks.analytical_model.tests.test_basic -v

# Run specific test method
python -c "
import unittest
from benchmarks.analytical_model.tests.test_basic import TestBasicFunctionality
suite = unittest.TestLoader().loadTestsFromName('test_hardware_registry', TestBasicFunctionality)
unittest.TextTestRunner(verbosity=2).run(suite)
"
```

### Test Coverage

The test suite covers:

- **Hardware Registry**: Hardware instantiation and specification validation
- **Model Registry**: Model instantiation and parameter validation  
- **Utilization Calculator**: Core MFU/MBU calculation logic
- **CSV Operations**: Parsing and output generation
- **End-to-End**: Complete analysis pipeline

### Adding Tests

```python
import unittest
from benchmarks.analytical_model import get_hardware, get_model

class TestNewFeature(unittest.TestCase):
    def test_new_functionality(self):
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues and Solutions

#### **"Hardware 'X' not found"**
```bash
# Check available hardware
python -m benchmarks.analytical_model --list-hardware

# Use exact name from list (lowercase)
python -m benchmarks.analytical_model --hardware a100  # not A100
```

#### **"CSV format is invalid"**
```bash
# Validate CSV format first
python -m benchmarks.analytical_model --validate-input --input your_file.csv

# Check for required columns
head -1 your_file.csv
```

#### **Very High MBU Values**
This is often normal - indicates theoretical model limitations. Consider:
- Memory access patterns not captured in analytical model
- Hardware cache effects reducing actual bandwidth requirements
- Optimized kernels with better memory efficiency

#### **Very Low MFU Values**
Possible causes:
- Memory-bound workload (check if MBU is high)
- Inefficient kernel utilization
- Suboptimal parallelism configuration
- Model architecture not well-suited to hardware

#### **Import Errors**
```bash
# Ensure you're in the vLLM root directory
cd /path/to/vllm
python -m benchmarks.analytical_model --help

# Check Python path if needed
export PYTHONPATH=/path/to/vllm:$PYTHONPATH
```

### Debug Mode

For detailed debugging information:

```bash
python -m benchmarks.analytical_model \
  --input benchmark_results.csv \
  --output analysis_results.csv \
  --hardware a100 \
  --model mixtral-8x7b \
  --verbose
```

### Performance Optimization Tips

1. **For Large CSV Files**: Process in batches if memory becomes an issue
2. **For Many Configurations**: Consider parallel processing for independent calculations
3. **For Custom Models**: Cache calculated parameters to avoid recomputation

## Contributing

To contribute to the analytical model:

1. **Follow the established patterns** for hardware and model implementations
2. **Add comprehensive tests** for new functionality  
3. **Update documentation** including this README
4. **Validate against known benchmarks** to ensure accuracy

## License

This analytical model is part of the vLLM project and follows the same license terms.