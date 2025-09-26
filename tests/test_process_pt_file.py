#!/usr/bin/env python3
"""
Test script to demonstrate processing an existing .pt file with protein sequences.
"""

import torch
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.download_real_uniref50 import RealUniRef50Downloader


def create_sample_pt_file(filename: str = "sample_proteins.pt"):
    """Create a sample .pt file with protein sequences for testing."""
    
    sample_data = [
        {'protein_seq': 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN'},
        {'protein_seq': 'MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV'},
        {'protein_seq': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'},
        {'protein_seq': 'MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE'},
        {'protein_seq': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'},
    ]
    
    # Save sample data
    torch.save(sample_data, filename)
    print(f"✅ Created sample .pt file: {filename}")
    print(f"📊 Contains {len(sample_data)} protein sequences")
    
    return filename


def test_process_pt_file():
    """Test processing a .pt file."""
    
    print("🧪 Testing .pt file processing...")
    
    # Create sample file
    sample_file = create_sample_pt_file()
    
    try:
        # Initialize downloader
        downloader = RealUniRef50Downloader(
            output_dir="./test_output",
            num_sequences=10  # Small number for testing
        )
        
        # Process the sample file
        output_file = downloader.process_from_file(
            input_file=sample_file,
            output_filename="test_processed_proteins.pt"
        )
        
        print(f"\n✅ Successfully processed file!")
        print(f"📁 Output: {output_file}")
        
        # Verify the output
        processed_data = torch.load(output_file)
        print(f"📊 Processed {len(processed_data)} sequences")
        
        # Show sample
        if processed_data:
            sample = processed_data[0]
            print(f"🔍 Sample processed entry:")
            print(f"   - Sequence: {sample['protein_seq'][:50]}...")
            print(f"   - Length: {sample['length']}")
            print(f"   - Token shape: {sample['prot_tokens'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if Path(sample_file).exists():
            Path(sample_file).unlink()
            print(f"🧹 Cleaned up sample file: {sample_file}")


def main():
    """Main test function."""
    print("🚀 Testing .pt file processing functionality")
    print("=" * 50)
    
    success = test_process_pt_file()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\n📖 Usage example:")
        print("python scripts/download_real_uniref50.py --input_file your_proteins.pt --output_dir ./output --num_sequences 1000")
    else:
        print("❌ Tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
