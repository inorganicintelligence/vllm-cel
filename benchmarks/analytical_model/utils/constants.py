"""Constants for analytical modeling."""

# Data type sizes in bytes
DTYPE_SIZES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'fp8': 1,
    'int8': 1,
    'int4': 0.5,
}

# Memory unit conversions
BYTES_TO_GB = 1024**3
BYTES_TO_MB = 1024**2
BYTES_TO_KB = 1024

# FLOPS unit conversions
FLOPS_TO_TFLOPS = 1e12
FLOPS_TO_GFLOPS = 1e9
FLOPS_TO_MFLOPS = 1e6

# Common model architectures
MODEL_ARCHITECTURES = {
    'mixtral-8x7b': {
        'type': 'moe',
        'experts': 8,
        'experts_per_token': 2,
    },
    'deepseek-v3': {
        'type': 'moe',
        'experts': 256,
        'experts_per_token': 8,
    },
}

# Parallelism defaults
DEFAULT_TP_SIZE = 1
DEFAULT_EP_ENABLED = False
DEFAULT_USE_FLASH_ATTENTION = True