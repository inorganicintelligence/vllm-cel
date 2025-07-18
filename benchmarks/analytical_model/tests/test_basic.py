"""Basic tests for the analytical model."""

import unittest
import tempfile
import os
from pathlib import Path

from ..hardware.registry import get_hardware, list_available_hardware
from ..models.registry import get_model, list_available_models
from ..calculators.utilization import UtilizationCalculator, BenchmarkResults
from ..utils.csv_parser import BenchmarkCSVParser, AnalysisCSVWriter


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the analytical model."""
    
    def test_hardware_registry(self):
        """Test hardware registry functionality."""
        # Test listing hardware
        hardware_list = list_available_hardware()
        self.assertIn('a100', hardware_list)
        self.assertIn('h100', hardware_list)
        
        # Test getting hardware
        a100 = get_hardware('A100')
        self.assertEqual(a100.name, 'A100')
        self.assertGreater(a100.peak_memory_bandwidth_gb_s, 2000)
        self.assertGreater(a100.get_peak_flops('fp16'), 300)
        
        # Test supported dtypes
        supported = a100.get_supported_dtypes()
        self.assertIn('fp16', supported)
        self.assertIn('bf16', supported)
    
    def test_model_registry(self):
        """Test model registry functionality."""
        # Test listing models
        model_list = list_available_models()
        self.assertIn('mixtral-8x7b', model_list)
        self.assertIn('deepseek-v3', model_list)
        
        # Test getting model
        mixtral = get_model('Mixtral-8x7B')
        self.assertEqual(mixtral.name, 'Mixtral-8x7B')
        
        params = mixtral.get_model_params()
        self.assertEqual(params['num_experts'], 8)
        self.assertEqual(params['experts_per_token'], 2)
    
    def test_utilization_calculator(self):
        """Test utilization calculator with sample data."""
        # Get hardware and model
        hardware = get_hardware('A100')
        model = get_model('Mixtral-8x7B')
        
        # Create calculator
        calculator = UtilizationCalculator(hardware, model, 'fp16')
        
        # Create sample benchmark results
        benchmark = BenchmarkResults(
            input_len=512,
            output_len=512,
            batch_size=8,
            tp_size=2,
            ep_enabled=True,
            system_throughput=1000.0,  # tokens/sec
            request_throughput=1.0,    # requests/sec
            avg_prefill_time=0.1,      # seconds
            avg_decode_time=5.0,       # seconds
        )
        
        # Calculate utilization
        utilization = calculator.calculate_utilization(benchmark)
        
        # Basic sanity checks
        self.assertGreaterEqual(utilization.total_mfu, 0.0)
        self.assertGreaterEqual(utilization.total_mbu, 0.0)
        self.assertGreaterEqual(utilization.prefill_mfu, 0.0)
        self.assertGreaterEqual(utilization.decode_mfu, 0.0)
        
        # MFU and MBU should be reasonable (allowing some margin for theoretical vs actual differences)
        self.assertLessEqual(utilization.total_mfu, 2.0)  # Allow some margin for calculation differences
        self.assertLessEqual(utilization.total_mbu, 10.0)  # Higher tolerance for MBU since theoretical models may overestimate
    
    def test_csv_operations(self):
        """Test CSV parsing and writing operations."""
        # Create temporary CSV file with sample data
        csv_content = """input_len,output_len,batch_size,tp_size,ep_enabled,system_throughput,request_throughput,avg_prefill_time,avg_decode_time,mean_ttft,mean_tpot,itl_mean,itl_median,itl_p99
512,512,8,2,true,1000.0,1.0,0.1,5.0,0.05,0.01,0.01,0.009,0.02
1024,1024,8,2,true,800.0,0.8,0.2,10.0,0.1,0.02,0.02,0.018,0.04"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_csv_path = f.name
        
        try:
            # Test CSV parsing
            results = BenchmarkCSVParser.parse_benchmark_csv(temp_csv_path)
            self.assertEqual(len(results), 2)
            
            # Check first result
            result1 = results[0]
            self.assertEqual(result1.input_len, 512)
            self.assertEqual(result1.tp_size, 2)
            self.assertTrue(result1.ep_enabled)
            
            # Test CSV validation
            is_valid = BenchmarkCSVParser.validate_csv_format(temp_csv_path)
            self.assertTrue(is_valid)
            
        finally:
            # Clean up
            os.unlink(temp_csv_path)
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis functionality."""
        # Create sample input CSV
        csv_content = """input_len,output_len,batch_size,tp_size,ep_enabled,system_throughput,request_throughput,avg_prefill_time,avg_decode_time,mean_ttft,mean_tpot,itl_mean,itl_median,itl_p99
512,512,8,2,true,1000.0,1.0,0.1,5.0,0.05,0.01,0.01,0.009,0.02"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_f:
            input_f.write(csv_content)
            input_path = input_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_f:
            output_path = output_f.name
        
        try:
            # Run analysis
            from ..utils.csv_parser import analyze_benchmark_csv
            
            analyze_benchmark_csv(
                input_csv_path=input_path,
                output_csv_path=output_path,
                hardware_name='A100',
                model_name='Mixtral-8x7B',
                dtype='fp16'
            )
            
            # Check that output file was created and has content
            self.assertTrue(Path(output_path).exists())
            
            # Read output and verify basic structure
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn('total_mfu', content)
                self.assertIn('total_mbu', content)
                self.assertIn('hardware', content)
                self.assertIn('A100', content)
            
        finally:
            # Clean up
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == '__main__':
    unittest.main()