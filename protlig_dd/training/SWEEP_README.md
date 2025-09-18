# 🚀 Super Lazy Wandb Hyperparameter Sweep

For the extremely lazy developer who wants to run hyperparameter sweeps with minimal effort.

## 🎯 Quick Start (Ultra Lazy Mode)

1. **Edit the paths in `lazy_sweep.sh`** (just the top variables):
   ```bash
   CONFIG="./configs/your_config.yaml"  # Point to your config file
   DATAFILE="./input_data/your_data.pt"  # Point to your data file
   ```

2. **Run the script**:
   ```bash
   cd protlig_dd/training
   ./lazy_sweep.sh
   ```

3. **Go grab coffee** ☕ and check your wandb dashboard!

## 🎛️ What It Does

- **Limits training to 2 epochs** for fast hyperparameter screening
- **Tests 20 different hyperparameter combinations** by default
- **Uses Bayesian optimization** to find good hyperparameters efficiently
- **Includes early termination** for obviously bad runs
- **Automatically creates unique run names**

## 🔧 Hyperparameters Being Tuned

- **Learning Rate**: 1e-5 to 1e-3 (log scale)
- **Batch Size**: 8, 16, 32
- **Model Size**: Hidden size, blocks, attention heads
- **Regularization**: Dropout, weight decay
- **Training Dynamics**: Warmup, accumulation steps, EMA
- **Noise Schedule**: Sigma min/max for diffusion
- **Curriculum Learning**: On/off with different schedules

## 🎨 Customization (If You're Not That Lazy)

### Modify Sweep Parameters
Edit `sweep_config.yaml` to change:
- Number of runs (`count`)
- Hyperparameter ranges
- Optimization method (bayes, grid, random)

### Manual Launch
```bash
python launch_sweep.py \
    --work_dir ./my_sweep \
    --config ./my_config.yaml \
    --datafile ./my_data.pt \
    --project my-sweep-project \
    --count 50
```

### Run Additional Agents
After creating a sweep, you can run multiple agents in parallel:
```bash
wandb agent YOUR_SWEEP_ID
```

## 📊 Results

Check your wandb dashboard at: https://wandb.ai/

The sweep will automatically:
- Track validation loss as the primary metric
- Log all hyperparameters and metrics
- Create parallel coordinate plots
- Show hyperparameter importance
- Generate optimization history

## 🚨 Notes

- Each run is only 2 epochs, so this is for **hyperparameter screening**, not final training
- Uses `force_fresh_start=True` so each run starts clean
- Uses `simple` sampling method for speed
- Creates a separate work directory for each sweep to avoid conflicts

## 🎉 After the Sweep

1. **Check the best runs** in wandb dashboard
2. **Pick the top 3-5 hyperparameter combinations**
3. **Run full training** (50+ epochs) with the best configs
4. **Profit!** 💰

---

*Made with ❤️ for lazy developers who want good results with minimal effort.*
