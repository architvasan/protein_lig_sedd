#!/usr/bin/env python3
"""
Fix and summarize evaluation results
"""

import json
import numpy as np
from collections import Counter
from pathlib import Path

def create_summary():
    """Create a clean summary of evaluation results."""
    
    # Manual summary based on the output
    results = {
        "evaluation_metadata": {
            "timestamp": "2025-09-16 15:48:00",
            "model_checkpoint": "best_checkpoint.pth",
            "training_step": 1000,
            "model_parameters": 66724096,
            "samples_generated": 50
        },
        "training_data_stats": {
            "sequences_analyzed": 352,
            "sequence_length": 512,
            "vocabulary_size": 25,
            "top_token_distribution": {
                "token_1": 30.6,
                "token_5": 5.2,
                "token_7": 4.0,
                "token_8": 4.5,
                "token_10": 4.7,
                "token_13": 4.0,
                "token_14": 6.2,
                "token_20": 5.9,
                "token_21": 4.5,
                "token_22": 4.5
            },
            "sequence_length_stats": {
                "mean": 512.0,
                "std": 0.0,
                "min": 512,
                "max": 512
            }
        },
        "generated_samples_stats": {
            "num_samples": 50,
            "sample_shape": [50, 256],
            "top_token_distribution": {
                "token_1": 57.6,
                "token_0": 1.7,
                "token_2": 1.8,
                "token_3": 0.8,
                "token_4": 1.5,
                "token_5": 0.9,
                "token_6": 0.9,
                "token_7": 1.1,
                "token_8": 1.0,
                "token_9": 1.1
            },
            "sequence_length_stats": {
                "mean": 245.4,
                "std": 4.6,
                "min": 232,
                "max": 253
            },
            "diversity_stats": {
                "unique_samples": 50,
                "total_samples": 50,
                "diversity_ratio": 1.000
            }
        },
        "quality_metrics": {
            "kl_divergence": 3.4338,
            "js_divergence": 0.1853,
            "wasserstein_distance": 260.0767
        },
        "token_distribution_comparison": {
            "token_1": {"train": 30.6, "generated": 57.6, "difference": 27.0},
            "token_8": {"train": 4.5, "generated": 1.0, "difference": -3.5},
            "token_10": {"train": 4.7, "generated": 1.0, "difference": -3.7},
            "token_13": {"train": 4.0, "generated": 1.1, "difference": -2.9},
            "token_14": {"train": 6.2, "generated": 1.1, "difference": -5.1},
            "token_19": {"train": 3.6, "generated": 1.0, "difference": -2.6},
            "token_20": {"train": 5.9, "generated": 0.9, "difference": -5.0},
            "token_21": {"train": 4.5, "generated": 1.0, "difference": -3.5},
            "token_22": {"train": 4.5, "generated": 1.0, "difference": -3.5},
            "token_23": {"train": 0.8, "generated": 0.9, "difference": 0.1}
        },
        "assessment": {
            "strengths": [
                "Perfect diversity (100% unique samples)",
                "Stable generation without failures",
                "Reasonable sequence lengths",
                "No mode collapse"
            ],
            "weaknesses": [
                "Heavy bias toward token 1 (57.6% vs 30.6%)",
                "Underutilization of vocabulary",
                "Significant distribution mismatch",
                "Shorter sequences than training data"
            ],
            "likely_causes": [
                "Insufficient training (only 1000 steps)",
                "Suboptimal sampling parameters",
                "Need for better noise scheduling"
            ],
            "recommendations": [
                "Continue training for 50,000+ steps",
                "Optimize sampling temperature and diffusion steps",
                "Generate more samples for better statistics",
                "Implement biological sequence validation"
            ]
        },
        "benchmark_comparison": {
            "kl_divergence": {"value": 3.4338, "benchmark": "Poor (>2.0)"},
            "js_divergence": {"value": 0.1853, "benchmark": "Acceptable (0.1-0.3)"},
            "diversity": {"value": 1.000, "benchmark": "Excellent (>0.9)"}
        }
    }
    
    # Save the clean results
    output_file = Path("evaluation_results") / "evaluation_summary_clean.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Clean evaluation results saved to: {output_file}")
    
    # Create a markdown summary
    markdown_summary = f"""# 🎯 SEDD Model Evaluation Summary

## 📊 **Quick Stats**
- **Model**: 66.7M parameters, trained for 1,000 steps
- **Samples**: 50 unique sequences generated (100% diversity)
- **Quality**: Mixed results - excellent diversity, poor distribution matching

## 🏆 **Key Metrics**
| Metric | Value | Assessment |
|--------|-------|------------|
| KL Divergence | 3.43 | Poor (>2.0) |
| JS Divergence | 0.19 | Acceptable (0.1-0.3) |
| Sample Diversity | 1.00 | Excellent (>0.9) |
| Unique Samples | 50/50 | Perfect |

## 🎯 **Main Findings**

### ✅ **Strengths**
- Perfect sample diversity (no duplicates)
- Stable generation process
- Model architecture working correctly

### ⚠️ **Issues**
- Heavy bias toward token 1 (57.6% vs 30.6% in training)
- Underutilization of vocabulary
- Shorter sequences than training data (245 vs 512 tokens)

## 💡 **Recommendations**
1. **Continue Training**: Current 1,000 steps → 50,000+ steps
2. **Optimize Sampling**: Adjust temperature and diffusion steps
3. **Extended Evaluation**: Generate 500+ samples for better statistics
4. **Biological Validation**: Check sequence validity

## 🚀 **Next Steps**
The model shows promise but needs significantly more training to achieve high-quality generation that matches the training distribution.
"""
    
    markdown_file = Path("evaluation_results") / "EVALUATION_SUMMARY.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_summary)
    
    print(f"✅ Markdown summary saved to: {markdown_file}")
    
    return results

if __name__ == "__main__":
    results = create_summary()
    print("\n🎉 Evaluation results processed successfully!")
    print(f"📁 Check evaluation_results/ directory for:")
    print(f"   - evaluation_summary_clean.json (structured data)")
    print(f"   - EVALUATION_SUMMARY.md (readable summary)")
    print(f"   - token_distribution.png (visualization)")
    print(f"   - sequence_lengths.png (visualization)")
