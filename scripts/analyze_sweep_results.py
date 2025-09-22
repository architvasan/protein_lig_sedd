#!/usr/bin/env python3
"""
Analyze hyperparameter sweep results from Wandb.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import argparse


def analyze_sweep_results(project_name: str = "uniref50_hyperparam_sweep", 
                         output_dir: str = "./sweep_analysis"):
    """
    Analyze sweep results from Wandb.
    
    Note: This is a template function. To use it, you'll need to:
    1. Install wandb: pip install wandb
    2. Login to wandb: wandb login
    3. Uncomment and modify the wandb code below
    """
    
    print("📊 HYPERPARAMETER SWEEP ANALYSIS")
    print("=" * 60)
    print(f"🔍 Project: {project_name}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Template for Wandb integration (uncomment and modify as needed)
    """
    import wandb
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get runs from the project
    runs = api.runs(project_name)
    
    # Extract data from runs
    data = []
    for run in runs:
        if run.state == "finished":  # Only analyze completed runs
            row = {
                'run_name': run.name,
                'run_id': run.id,
                'state': run.state,
                'duration': run.duration,
            }
            
            # Add config parameters
            for key, value in run.config.items():
                row[f'config_{key}'] = value
            
            # Add summary metrics
            for key, value in run.summary.items():
                if isinstance(value, (int, float)):
                    row[f'metric_{key}'] = value
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("❌ No completed runs found!")
        return
    
    print(f"✅ Found {len(df)} completed runs")
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, "sweep_results.csv"), index=False)
    
    # Analysis and visualization
    analyze_hyperparameter_importance(df, output_dir)
    create_performance_plots(df, output_dir)
    find_best_configurations(df, output_dir)
    """
    
    # For now, create a template analysis
    create_analysis_template(output_dir)


def create_analysis_template(output_dir: str):
    """Create template analysis files."""
    
    # Create example analysis notebook
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UniRef50 Hyperparameter Sweep Analysis\n",
                    "\n",
                    "This notebook analyzes the results of hyperparameter sweeps for UniRef50 SEDD training.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import wandb\n",
                    "\n",
                    "# Initialize wandb API\n",
                    "api = wandb.Api()\n",
                    "project_name = 'uniref50_hyperparam_sweep'\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load sweep results\n",
                    "runs = api.runs(project_name)\n",
                    "\n",
                    "data = []\n",
                    "for run in runs:\n",
                    "    if run.state == 'finished':\n",
                    "        row = {'name': run.name, 'id': run.id}\n",
                    "        row.update(run.config)\n",
                    "        row.update(run.summary)\n",
                    "        data.append(row)\n",
                    "\n",
                    "df = pd.DataFrame(data)\n",
                    "print(f'Found {len(df)} completed runs')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Analyze key metrics\n",
                    "key_metrics = ['val_loss', 'train_loss', 'generation_quality']\n",
                    "key_params = ['learning_rate', 'batch_size', 'hidden_size', 'n_heads']\n",
                    "\n",
                    "# Correlation analysis\n",
                    "correlation_data = df[key_params + key_metrics].corr()\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)\n",
                    "plt.title('Hyperparameter-Metric Correlations')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Find best configurations\n",
                    "best_runs = df.nsmallest(5, 'val_loss')\n",
                    "print('Top 5 configurations by validation loss:')\n",
                    "print(best_runs[['name'] + key_params + key_metrics])"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_path = os.path.join(output_dir, "sweep_analysis.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    # Create analysis script template
    script_content = '''#!/usr/bin/env python3
"""
Hyperparameter sweep analysis script.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sweep(project_name="uniref50_hyperparam_sweep"):
    """Analyze hyperparameter sweep results."""
    
    # Initialize wandb API
    api = wandb.Api()
    runs = api.runs(project_name)
    
    # Extract data
    data = []
    for run in runs:
        if run.state == "finished":
            row = {"name": run.name, "id": run.id}
            row.update(run.config)
            row.update(run.summary)
            data.append(row)
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No completed runs found!")
        return
    
    print(f"Analyzing {len(df)} completed runs")
    
    # Key metrics to analyze
    metrics = ["val_loss", "train_loss", "quick_gen/success_rate"]
    params = ["learning_rate", "batch_size", "hidden_size", "n_heads", "dropout"]
    
    # Find best configurations
    best_runs = df.nsmallest(5, "val_loss")
    print("\\nTop 5 configurations:")
    print(best_runs[["name"] + params + metrics])
    
    # Parameter importance analysis
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 8))
            for i, param in enumerate(params):
                if param in df.columns:
                    plt.subplot(2, 3, i+1)
                    plt.scatter(df[param], df[metric], alpha=0.6)
                    plt.xlabel(param)
                    plt.ylabel(metric)
                    plt.title(f"{metric} vs {param}")
            
            plt.tight_layout()
            plt.savefig(f"parameter_analysis_{metric.replace('/', '_')}.png")
            plt.show()

if __name__ == "__main__":
    analyze_sweep()
'''
    
    script_path = os.path.join(output_dir, "analyze_results.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Create README
    readme_content = """# Hyperparameter Sweep Analysis

This directory contains tools for analyzing hyperparameter sweep results.

## Files

- `sweep_analysis.ipynb`: Jupyter notebook for interactive analysis
- `analyze_results.py`: Python script for automated analysis
- `sweep_results.csv`: Raw results data (generated after running analysis)

## Usage

### Prerequisites

```bash
pip install wandb pandas matplotlib seaborn jupyter
wandb login
```

### Running Analysis

1. **Interactive Analysis (Recommended)**:
   ```bash
   jupyter notebook sweep_analysis.ipynb
   ```

2. **Automated Analysis**:
   ```bash
   python analyze_results.py
   ```

### Key Metrics to Monitor

- **val_loss**: Validation loss (lower is better)
- **train_loss**: Training loss (lower is better)
- **quick_gen/success_rate**: Generation success rate (higher is better)
- **quick_gen/avg_length**: Average generated sequence length
- **quick_gen/avg_unique_aa**: Average amino acid diversity

### Key Hyperparameters

- **learning_rate**: Optimizer learning rate
- **batch_size**: Training batch size
- **hidden_size**: Model hidden dimension
- **n_heads**: Number of attention heads
- **dropout**: Dropout rate
- **sampling_method**: Sampling method (rigorous/simple)

## Analysis Workflow

1. Load completed runs from Wandb
2. Extract configuration and metrics
3. Identify best performing configurations
4. Analyze parameter importance
5. Create visualizations
6. Generate recommendations

## Tips

- Focus on validation loss as primary metric
- Consider generation quality metrics for practical performance
- Look for parameter combinations that work well together
- Check for overfitting (train vs validation loss)
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created analysis templates in: {output_dir}")
    print(f"📓 Jupyter notebook: {notebook_path}")
    print(f"🐍 Python script: {script_path}")
    print(f"📖 README: {readme_path}")
    print()
    print("📋 Next steps:")
    print("1. Install required packages: pip install wandb pandas matplotlib seaborn jupyter")
    print("2. Login to wandb: wandb login")
    print("3. Run analysis: jupyter notebook sweep_analysis.ipynb")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("--project", type=str, default="uniref50_hyperparam_sweep",
                       help="Wandb project name")
    parser.add_argument("--output_dir", type=str, default="./sweep_analysis",
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    analyze_sweep_results(args.project, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
