# go1_arm_lab

## Run the Code in Isaac Lab v2.2.1 (Oct, 2025)
Train
```
python scripts/local_rsl/train.py --task=Isaac-Flat-widowgo1 --headless --logger=wandb
```
Play
```
python scripts/local_rsl/play.py --task=Isaac-Flat-widowgo1-Play
```
## 📦 Installation

1. Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to install IsaacLab v2.2.1.  
2. Clone this repository:
   ```
   git clone https://github.com/kabirraymalik/go1_arm_lab.git
   ```
3. Install the package using the Python interpreter that IsaacLab uses:
   ```
   # In your IsaacLab environment
   conda activate your_isaaclab_env
   
   # install project as package
   python -m pip install -e source/go1_arm_lab
   ```

### ⚙️ Training & Inference

#### Training

Run reinforcement-learning training with --headless for efficiency:

```
# Activate IsaacLab environment
conda activate your_isaaclab_env

# Go to go1_arm_lab
cd /path/to/go1_arm_lab

# Launch training (headless)
python scripts/rsl_rl/train.py --task Isaac-widowgo1-flat --headless
python scripts/rsl_rl/train.py --task Isaac-widowgo1-rough --headless
```

#### Inference

Deploy a trained policy in a single environment:

```
# Activate IsaacLab environment
conda activate your_isaaclab_env

# Go to go1_arm_lab
cd /path/to/go1_arm_lab

# Run inference
python scripts/rsl_rl/play.py --task Isaac-widowgo1-flat-play --num_envs 1
python scripts/rsl_rl/play.py --task Isaac-widowgo1-rough-play --num_envs 1
```

#### Sim2Sim

For details on sim2sim with this robot, plase see [widowgo1_sim2sim](https://github.com/kabirraymalik/widowgo1_sim2sim)

## Acknowledgments
This project was heavily structured and based off of [Go2Arm_Lab](https://github.com/zzzJie-Robot/Go2Arm_Lab)

Additional references and rewards based off of [robot_lab](https://github.com/fan-ziqi/robot_lab)

The RL algorithm implementation in this project references the [Deep-Whole-Body-Control](https://github.com/MarkFzp/Deep-Whole-Body-Control) project, for which we extend our sincere gratitude.