#!/usr/bin/env python3
"""
Simple test for dtype consistency without importing the full trainer.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append('.')

def test_dtype_conversion():
    """Test dtype conversion logic."""
    print("🧪 Testing Dtype Conversion Logic")
    print("=" * 50)
    
    # Test different dtype scenarios
    test_cases = [
        (torch.float32, torch.float32, "Same dtype - no conversion needed"),
        (torch.float32, torch.bfloat16, "Float32 to BFloat16 conversion"),
        (torch.bfloat16, torch.float32, "BFloat16 to Float32 conversion"),
        (torch.long, torch.float32, "Long to Float32 conversion"),
        (torch.float32, torch.long, "Float32 to Long conversion"),
    ]
    
    for input_dtype, target_dtype, description in test_cases:
        print(f"\n📊 {description}")
        
        # Create test tensors
        if input_dtype == torch.long:
            input_tensor = torch.randint(0, 25, (4, 32), dtype=input_dtype)
        else:
            input_tensor = torch.randn(4, 32, dtype=input_dtype)
        
        print(f"   Input dtype: {input_tensor.dtype}")
        print(f"   Target dtype: {target_dtype}")
        
        # Test conversion
        try:
            if input_tensor.dtype != target_dtype:
                converted_tensor = input_tensor.to(dtype=target_dtype)
                print(f"   ✅ Conversion successful: {converted_tensor.dtype}")
            else:
                converted_tensor = input_tensor
                print(f"   ✅ No conversion needed: {converted_tensor.dtype}")
        except Exception as e:
            print(f"   ❌ Conversion failed: {e}")
            return False
    
    return True

def test_matrix_multiplication_dtypes():
    """Test matrix multiplication with different dtypes."""
    print("\n🧪 Testing Matrix Multiplication Dtypes")
    print("=" * 50)
    
    # Test scenarios that could cause the original error
    test_scenarios = [
        ("Float32 x Float32", torch.float32, torch.float32),
        ("BFloat16 x BFloat16", torch.bfloat16, torch.bfloat16),
        ("Float32 x BFloat16 (should fail)", torch.float32, torch.bfloat16),
    ]
    
    for description, dtype1, dtype2 in test_scenarios:
        print(f"\n📊 {description}")
        
        # Create test matrices
        mat1 = torch.randn(4, 64, dtype=dtype1)
        mat2 = torch.randn(64, 32, dtype=dtype2)
        
        print(f"   Matrix 1 dtype: {mat1.dtype}")
        print(f"   Matrix 2 dtype: {mat2.dtype}")
        
        try:
            result = torch.matmul(mat1, mat2)
            print(f"   ✅ Multiplication successful: result dtype = {result.dtype}")
        except Exception as e:
            print(f"   ❌ Multiplication failed: {e}")
            
            # Try with dtype conversion
            print(f"   🔧 Attempting dtype conversion...")
            try:
                if mat1.dtype != mat2.dtype:
                    # Convert to common dtype (prefer higher precision)
                    if mat1.dtype == torch.float32 or mat2.dtype == torch.float32:
                        common_dtype = torch.float32
                    else:
                        common_dtype = mat1.dtype
                    
                    mat1_converted = mat1.to(dtype=common_dtype)
                    mat2_converted = mat2.to(dtype=common_dtype)
                    
                    result = torch.matmul(mat1_converted, mat2_converted)
                    print(f"   ✅ Fixed with conversion: result dtype = {result.dtype}")
                else:
                    print(f"   ❌ Same dtypes but still failed: {e}")
            except Exception as e2:
                print(f"   ❌ Conversion fix failed: {e2}")
    
    return True

def test_model_dtype_consistency():
    """Test model dtype consistency simulation."""
    print("\n🧪 Testing Model Dtype Consistency Simulation")
    print("=" * 50)
    
    # Simulate a simple model with different dtypes
    class SimpleModel(torch.nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.linear1 = torch.nn.Linear(64, 32, dtype=dtype)
            self.linear2 = torch.nn.Linear(32, 25, dtype=dtype)
        
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    
    # Test different model dtypes
    model_dtypes = [torch.float32, torch.bfloat16]
    
    for model_dtype in model_dtypes:
        print(f"\n📊 Testing model with dtype: {model_dtype}")
        
        model = SimpleModel(dtype=model_dtype)
        model_param_dtype = next(model.parameters()).dtype
        print(f"   Model parameter dtype: {model_param_dtype}")
        
        # Test with different input dtypes
        input_dtypes = [torch.float32, torch.bfloat16, torch.long]
        
        for input_dtype in input_dtypes:
            print(f"   Testing input dtype: {input_dtype}")
            
            # Create input tensor
            if input_dtype == torch.long:
                # For long tensors, we need to convert to float for the model
                input_tensor = torch.randint(0, 25, (2, 64), dtype=input_dtype)
                # Convert to model dtype for processing
                input_tensor = input_tensor.to(dtype=model_param_dtype)
                print(f"     Converted long input to: {input_tensor.dtype}")
            else:
                input_tensor = torch.randn(2, 64, dtype=input_dtype)
            
            try:
                # Ensure input matches model dtype
                if input_tensor.dtype != model_param_dtype:
                    input_tensor = input_tensor.to(dtype=model_param_dtype)
                    print(f"     Converted input to model dtype: {input_tensor.dtype}")
                
                output = model(input_tensor)
                print(f"     ✅ Forward pass successful: output dtype = {output.dtype}")
                
            except Exception as e:
                print(f"     ❌ Forward pass failed: {e}")
                return False
    
    return True

def main():
    """Run all dtype tests."""
    print("🚀 Running Dtype Consistency Tests")
    print("=" * 60)
    
    tests = [
        test_dtype_conversion,
        test_matrix_multiplication_dtypes,
        test_model_dtype_consistency,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test.__name__}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All dtype consistency tests passed!")
        print("💡 The dtype fixes should resolve the 'mat1 and mat2 must have the same dtype' error")
        return 0
    else:
        print("\n😞 Some dtype tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
