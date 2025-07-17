# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for Large Language Models (LLMs). It's designed for fast model execution with features like PagedAttention, continuous batching, and optimized CUDA kernels.

Key features:
- **PagedAttention**: Efficient memory management for attention key-value storage
- **Continuous batching**: Dynamic batching of incoming requests
- **Multi-backend support**: CUDA, ROCm, CPU, TPU, Intel XPU, AWS Neuron
- **Quantization**: Support for GPTQ, AWQ, AutoRound, INT4/8, FP8
- **Distributed inference**: Tensor and pipeline parallelism
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API

## Development Setup

### Building from source
```bash
# Install build dependencies
pip install -r requirements/build.txt

# Build vLLM (uses CMake for C++/CUDA extensions)
pip install -e .

# Or for development with CUDA
VLLM_USE_PRECOMPILED_KERNELS=0 pip install -e .
```

### Environment variables
- `VLLM_TARGET_DEVICE`: Target device (cuda/rocm/cpu/tpu/neuron/xpu)
- `VLLM_USE_PRECOMPILED_KERNELS`: Whether to use precompiled kernels (default: 1)
- `MAX_JOBS`: Number of parallel compilation jobs

## Code Linting and Formatting

vLLM uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pip install -r requirements/lint.txt
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Tools used:
- **yapf**: Python formatting
- **ruff**: Python linting and import sorting  
- **mypy**: Type checking
- **clang-format**: C++/CUDA formatting
- **isort**: Import sorting

## Testing

### Running tests
```bash
# Install test dependencies
pip install -r requirements/test.txt

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/basic_correctness/
pytest tests/models/
pytest tests/distributed/

# Run single test file
pytest tests/test_sampling_params.py

# Run with specific markers
pytest -m "not skip_global_cleanup"
pytest -m "core_model"
```

### Test configuration
- Test markers defined in `pyproject.toml`
- Distributed tests require multiple GPUs
- Some tests marked as optional (use `--optional` to run)

## Architecture Overview

### Core Components

1. **Entrypoints**:
   - `LLM`: Offline inference interface (`vllm/entrypoints/llm.py`)
   - `vllm serve`: OpenAI-compatible API server (`vllm/entrypoints/openai/api_server.py`)

2. **Engine Layer**:
   - `LLMEngine`: Core synchronous engine (`vllm/engine/llm_engine.py`)
   - `AsyncLLMEngine`: Asynchronous wrapper for serving (`vllm/engine/async_llm_engine.py`)

3. **Execution Layer**:
   - **Worker**: Process managing one accelerator device
   - **ModelRunner**: Loads and runs model within worker  
   - **Model**: Actual `torch.nn.Module` instance

4. **Core Systems**:
   - **Scheduler**: Request batching and scheduling (`vllm/core/scheduler.py`)
   - **Attention**: PagedAttention implementation (`vllm/attention/`)
   - **Memory Management**: KV cache and block management (`vllm/core/block_manager.py`)

### Directory Structure

- `vllm/`: Main Python package
  - `engine/`: LLM engines and configuration
  - `entrypoints/`: User-facing APIs (LLM class, CLI, OpenAI server)
  - `core/`: Core scheduling and memory management
  - `attention/`: Attention mechanisms and kernels
  - `model_executor/`: Model loading and execution
  - `distributed/`: Multi-GPU/node communication
  - `compilation/`: Torch compilation support
- `csrc/`: C++/CUDA kernels and extensions
- `tests/`: Comprehensive test suite
- `benchmarks/`: Performance benchmarking tools
- `examples/`: Usage examples and demos
- `docs/`: Documentation

### Key Concepts

- **VllmConfig**: Central configuration object passed throughout system
- **Continuous Batching**: Dynamic request batching for higher throughput
- **PagedAttention**: Virtual memory-style attention to reduce fragmentation
- **Chunked Prefill**: Processing long contexts in chunks
- **Speculative Decoding**: Using draft models to accelerate generation

## Common Development Tasks

### Adding new models
1. Create model class in `vllm/model_executor/models/`
2. Register in `vllm/model_executor/models/__init__.py`
3. Follow uniform constructor signature: `__init__(self, *, vllm_config: VllmConfig, prefix: str = "")`
4. Add tests in `tests/models/`

### Custom kernels
- CUDA kernels in `csrc/`
- Build system uses CMake
- Register in `vllm/_C.py` bindings

### Quantization support
- Implement in `vllm/model_executor/layers/quantization/`
- Add config class and weight loading logic
- Support sharding during initialization

## Platform Support

- **CUDA**: Primary platform (NVIDIA GPUs)
- **ROCm**: AMD GPU support  
- **CPU**: x86, ARM, PowerPC architectures
- **TPU**: Google Cloud TPU support
- **Intel**: CPU and XPU (GPU) support
- **AWS Neuron**: Inferentia/Trainium support

## API Usage Examples

### Offline inference
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

### OpenAI-compatible server
```bash
vllm serve facebook/opt-125m --port 8000
# Use with OpenAI client libraries
```

## Performance Considerations

- Use tensor parallelism for large models
- Enable prefix caching for repeated prompts  
- Consider chunked prefill for long contexts
- Use appropriate quantization for memory constraints
- Profile with built-in tools in `vllm/entrypoints/`

## Claude code planning and execution best practices
1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.