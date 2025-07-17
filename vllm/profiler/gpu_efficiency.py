"""GPU efficiency metrics collection and calculation utilities."""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
import logging
from contextlib import contextmanager

from vllm.profiler.hardware_specs import get_gpu_specs, calculate_mfu, calculate_mbu

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors GPU utilization and memory bandwidth during benchmark execution."""
    
    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        """Initialize GPU monitor.
        
        Args:
            device_id: GPU device ID to monitor
            sample_interval: Sampling interval in seconds
        """
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Collected data
        self.gpu_utilization_samples: List[float] = []
        self.memory_utilization_samples: List[float] = []
        self.memory_usage_samples: List[float] = []
        self.power_samples: List[float] = []
        self.timestamps: List[float] = []
        
        # GPU info
        self.gpu_name: Optional[str] = None
        self.gpu_specs: Optional[Tuple[float, float, float]] = None
        self.memory_total: Optional[int] = None
        
        self._initialize_nvml()
    
    def _initialize_nvml(self):
        """Initialize NVML and get GPU information."""
        if not NVML_AVAILABLE:
            logger.warning("NVML not available, GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            # Get GPU name and specs
            self.gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            self.gpu_specs = get_gpu_specs(self.gpu_name)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.memory_total = mem_info.total
            
            logger.info(f"Initialized GPU monitor for {self.gpu_name}")
            if self.gpu_specs:
                peak_flops, peak_bw, peak_mem = self.gpu_specs
                logger.info(f"GPU specs: {peak_flops:.1f} TFLOP/s, {peak_bw:.1f} GB/s, {peak_mem:.1f} GB")
            else:
                logger.warning(f"GPU specs not found for {self.gpu_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_name = None
            self.gpu_specs = None
    
    def _monitor_loop(self):
        """Main monitoring loop that collects GPU metrics."""
        if not NVML_AVAILABLE:
            return
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            while self.monitoring:
                start_time = time.time()
                
                try:
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization_samples.append(util.gpu)
                    self.memory_utilization_samples.append(util.memory)
                    
                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_gb = mem_info.used / (1024**3)
                    self.memory_usage_samples.append(memory_used_gb)
                    
                    # Power consumption
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        self.power_samples.append(power)
                    except:
                        self.power_samples.append(0.0)
                    
                    self.timestamps.append(start_time)
                    
                except Exception as e:
                    logger.debug(f"Error collecting GPU metrics: {e}")
                
                # Sleep for the remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sample_interval - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"GPU monitoring loop failed: {e}")
    
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Started GPU monitoring")
    
    def stop_monitoring(self):
        """Stop GPU monitoring and return collected metrics."""
        if not self.monitoring:
            return {}
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug("Stopped GPU monitoring")
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected GPU metrics."""
        if not self.gpu_utilization_samples:
            return {}
        
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0
        
        def safe_max(values):
            return max(values) if values else 0.0
        
        metrics = {
            "gpu_name": self.gpu_name,
            "gpu_utilization_percent": safe_mean(self.gpu_utilization_samples),
            "memory_utilization_percent": safe_mean(self.memory_utilization_samples),
            "peak_memory_usage_gb": safe_max(self.memory_usage_samples),
            "avg_power_watts": safe_mean(self.power_samples),
            "sample_count": len(self.gpu_utilization_samples),
        }
        
        # Add GPU specs if available
        if self.gpu_specs:
            peak_flops, peak_bw, peak_mem = self.gpu_specs
            metrics.update({
                "gpu_peak_flops_tflops": peak_flops,
                "gpu_peak_bandwidth_gb_s": peak_bw,
                "gpu_total_memory_gb": peak_mem,
            })
        
        return metrics
    
    def clear_samples(self):
        """Clear all collected samples."""
        self.gpu_utilization_samples.clear()
        self.memory_utilization_samples.clear()
        self.memory_usage_samples.clear()
        self.power_samples.clear()
        self.timestamps.clear()


@contextmanager
def gpu_monitor_context(device_id: int = 0, sample_interval: float = 0.1):
    """Context manager for GPU monitoring during benchmark execution.
    
    Args:
        device_id: GPU device ID to monitor
        sample_interval: Sampling interval in seconds
        
    Yields:
        GPUMonitor instance
    """
    monitor = GPUMonitor(device_id=device_id, sample_interval=sample_interval)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def calculate_efficiency_metrics(
    system_throughput: float,
    total_tokens: int,
    total_time: float,
    gpu_metrics: Dict[str, Any],
    model_config: Optional[Dict] = None
) -> Dict[str, float]:
    """Calculate MFU and MBU from benchmark results and GPU monitoring data.
    
    Args:
        system_throughput: System throughput in tokens/second
        total_tokens: Total tokens generated
        total_time: Total time in seconds
        gpu_metrics: GPU monitoring metrics from GPUMonitor
        model_config: Model configuration for FLOP calculation
        
    Returns:
        Dictionary with efficiency metrics
    """
    efficiency_metrics = {}
    
    # Extract GPU specs
    gpu_name = gpu_metrics.get("gpu_name")
    if not gpu_name:
        logger.warning("GPU name not available, cannot calculate efficiency metrics")
        return efficiency_metrics
    
    gpu_specs = get_gpu_specs(gpu_name)
    if not gpu_specs:
        logger.warning(f"GPU specs not found for {gpu_name}")
        return efficiency_metrics
    
    peak_flops_tflops, peak_bandwidth_gb_s, peak_memory_gb = gpu_specs
    
    # Basic efficiency metrics
    efficiency_metrics.update({
        "gpu_utilization_percent": gpu_metrics.get("gpu_utilization_percent", 0.0),
        "memory_utilization_percent": gpu_metrics.get("memory_utilization_percent", 0.0),
        "peak_memory_usage_gb": gpu_metrics.get("peak_memory_usage_gb", 0.0),
    })
    
    # Calculate MFU if model config available
    if model_config and system_throughput > 0:
        try:
            mfu = calculate_mfu(system_throughput, gpu_specs, model_config)
            efficiency_metrics["mfu_percent"] = mfu
            logger.debug(f"Calculated MFU: {mfu:.2f}%")
        except Exception as e:
            logger.warning(f"Failed to calculate MFU: {e}")
            efficiency_metrics["mfu_percent"] = 0.0
    else:
        efficiency_metrics["mfu_percent"] = 0.0
    
    # Estimate MBU from memory utilization
    # This is an approximation since we don't have direct memory throughput measurements
    memory_util_percent = gpu_metrics.get("memory_utilization_percent", 0.0)
    estimated_memory_throughput = (memory_util_percent / 100.0) * peak_bandwidth_gb_s
    
    try:
        mbu = calculate_mbu(estimated_memory_throughput, gpu_specs)
        efficiency_metrics["mbu_percent"] = mbu
        logger.debug(f"Estimated MBU: {mbu:.2f}%")
    except Exception as e:
        logger.warning(f"Failed to calculate MBU: {e}")
        efficiency_metrics["mbu_percent"] = 0.0
    
    return efficiency_metrics


def get_model_config_from_engine(engine) -> Optional[Dict]:
    """Extract model configuration parameters from vLLM engine for FLOP calculation.
    
    Args:
        engine: vLLM engine instance
        
    Returns:
        Dictionary with model parameters or None if extraction fails
    """
    try:
        model_config = engine.model_config
        scheduler_config = engine.scheduler_config
        
        config = {
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_attention_heads,
            "vocab_size": model_config.vocab_size,
            "batch_size": scheduler_config.max_num_batched_tokens // scheduler_config.max_model_len,
            "seq_len": scheduler_config.max_model_len,
        }
        
        # Try to get intermediate size
        if hasattr(model_config, 'intermediate_size'):
            config["intermediate_size"] = model_config.intermediate_size
        
        logger.debug(f"Extracted model config: {config}")
        return config
        
    except Exception as e:
        logger.warning(f"Failed to extract model config: {e}")
        return None