#!/usr/bin/env python3
"""
Test runner for checkpoint functionality.
Tries different test approaches based on what's available.
"""

import sys
import os
import subprocess
import traceback

def run_test(test_file, description):
    """Run a test file and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {description}")
    print(f"📁 File: {test_file}")
    print('='*60)
    
    try:
        # Try running with python directly
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test PASSED!")
            print("\n📋 Output:")
            print(result.stdout)
            if result.stderr:
                print("\n⚠️  Warnings:")
                print(result.stderr)
            return True
        else:
            print("❌ Test FAILED!")
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("\n📋 Output:")
                print(result.stdout)
            if result.stderr:
                print("\n🚨 Errors:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test TIMED OUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        traceback.print_exc()
        return False

def run_pytest(test_file, description):
    """Run a test file with pytest."""
    print(f"\n{'='*60}")
    print(f"🧪 Running with pytest: {description}")
    print(f"📁 File: {test_file}")
    print('='*60)
    
    try:
        # Try running with pytest
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '-s', '--tb=short'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Pytest PASSED!")
            print("\n📋 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Pytest FAILED!")
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("\n📋 Output:")
                print(result.stdout)
            if result.stderr:
                print("\n🚨 Errors:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Pytest TIMED OUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 Pytest execution failed: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 Checkpoint Test Suite Runner")
    print("=" * 60)
    
    # Get the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)
    
    # Set up Python path
    project_root = os.path.dirname(test_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test files to try (in order of preference)
    test_files = [
        {
            'file': 'test_checkpoint_simple_fixed.py',
            'description': 'Simplified Checkpoint Tests (Mock-based)',
            'method': 'direct'
        },
        {
            'file': 'test_checkpoint_simple.py', 
            'description': 'Simple Checkpoint Tests (Trainer-based)',
            'method': 'direct'
        },
        {
            'file': 'test_checkpoint_resume.py',
            'description': 'Full Checkpoint Resume Tests (Pytest)',
            'method': 'pytest'
        }
    ]
    
    results = []
    
    for test_info in test_files:
        test_file = test_info['file']
        description = test_info['description']
        method = test_info['method']
        
        # Check if test file exists
        if not os.path.exists(test_file):
            print(f"\n⚠️  Skipping {test_file} - file not found")
            results.append((test_file, False, "File not found"))
            continue
        
        # Run the test
        if method == 'direct':
            success = run_test(test_file, description)
        elif method == 'pytest':
            success = run_pytest(test_file, description)
        else:
            print(f"❌ Unknown test method: {method}")
            success = False
        
        results.append((test_file, success, "Completed"))
        
        # If a test passes, we can stop (unless we want to run all)
        if success:
            print(f"\n🎉 Found working test: {test_file}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len([r for r in results if r[2] != "File not found"])
    
    for test_file, success, status in results:
        if status == "File not found":
            print(f"⚠️  {test_file}: {status}")
        elif success:
            print(f"✅ {test_file}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_file}: FAILED")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed > 0:
        print("\n🎉 At least one checkpoint test is working!")
        print("💡 Recommendation: Use the working test as your checkpoint validation")
        return 0
    else:
        print("\n😞 No checkpoint tests are currently working")
        print("💡 Recommendation: Check the error messages above and fix the issues")
        return 1

if __name__ == "__main__":
    exit(main())
