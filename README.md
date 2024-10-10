#  AIAgentplaying_ATARI

This project implements Deep Q-Network (DQN) agents for playing Atari games using reinforcement learning. The repository contains different versions and modifications of DQN agents such as vanilla DQN, Duel DQN, and Episodic DQN.


## Project Structure

```bash
.
├── agent.py                    # Core implementation of the DQN agent
├── agent_dqn_duel.py            # Dueling DQN implementation
├── agent_dqn.py                 # Vanilla DQN implementation
├── agent_dqn_epi.py             # Episodic DQN variant
├── argument.py                  # Command line argument parsing
├── atari_wrapper.py             # Environment wrappers for Atari games
├── dqn_model.py                 # Definition of the neural network model for DQN
├── environment.py               # Atari environment setup and initialization
├── main.py                      # Main script to train and run the DQN agents
├── test.py                      # Script for testing the trained DQN agent
└── README.md                    # Project overview and setup instructions
```

Here's how you can expand your README.md to include the setup instructions, environment configuration, and how to run your project, along with the goal of your implementation:

---

# CS595 - DQN_ATARI

This project implements Deep Q-Network (DQN) agents to play Atari games using reinforcement learning. The main goal is to train an agent to achieve a high score in Breakout using PyTorch and OpenAI's Gymnasium framework.

## Project Structure

```bash
.
├── agent.py                    # Core implementation of the DQN agent
├── agent_dqn_duel.py            # Dueling DQN implementation
├── agent_dqn.py                 # Vanilla DQN implementation
├── agent_dqn_epi.py             # Episodic DQN variant
├── argument.py                  # Command line argument parsing
├── atari_wrapper.py             # Environment wrappers for Atari games
├── dqn_model.py                 # Definition of the neural network model for DQN
├── environment.py               # Atari environment setup and initialization
├── main.py                      # Main script to train and run the DQN agents
├── test.py                      # Script for testing the trained DQN agent
└── README.md                    # Project overview and setup instructions
```

## Setup

### Recommended IDE
- Visual Studio Code (VS Code) is recommended for this project. [Install VS Code](https://code.visualstudio.com/).

### Steps

1. **Install Miniconda**: Follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html) to install Miniconda.

2. **Create a virtual environment**:  
   Run the following to create a virtual environment with Python 3.11.4:
   ```bash
   conda create -n myenv python=3.11.4
   ```
   This creates an environment named `myenv`. Ensure that Gymnasium supports Python versions 3.8, 3.9, 3.10, or 3.11 on Linux and macOS.

3. **Activate your virtual environment**:  
   ```bash
   conda activate myenv
   ```

4. **Install Gymnasium**:  
   Install the necessary libraries:
   ```bash
   pip install opencv-python-headless gymnasium[atari] autorom[accept-rom-license]
   ```

5. **Install PyTorch**:  
   Install PyTorch based on your system configuration by following the instructions [here](https://pytorch.org/get-started/locally/).  
   Alternatively, you can install PyTorch via pip:
   ```bash
   pip install torch torchvision torchaudio
   ```

6. **Install Ray for Atari wrapper**:  
   Install Ray for reinforcement learning and additional packages:
   ```bash
   pip install -U "ray[rllib]" ipywidgets
   ```

7. **Install additional dependencies**:  
   For successful execution of the code, install the following:
   ```bash
   pip install --upgrade scipy numpy
   ```

8. **Install video recording dependencies**:  
   To enable video recording during testing, install:
   ```bash
   pip install moviepy ffmpeg
   ```

9. **Install tqdm for progress visualization**:  
   For nice terminal output during testing:
   ```bash
   pip install tqdm
   ```

## How to Run

### Training DQN:
To start training a DQN agent, run:
```bash
python main.py --train_dqn
```

### Testing DQN:
To test the performance of the trained DQN agent:
```bash
python main.py --test_dqn
```

### Testing DQN with video recording:
For testing with video recording (this may slow down the execution, so it's recommended for small numbers of episodes):
```bash
python main.py --test_dqn --record_video
```

## Goal

In this project, the goal is to implement Deep Q-Network (DQN) to play the Atari game **Breakout**. The task is to train the agent to achieve an average reward of over **40 points** across 100 episodes. Each episode consists of 5 lives. The training must use OpenAI's Atari wrapper and a clipped reward system.

## Requirements

- Python 3.8, 3.9, 3.10, or 3.11
- Gymnasium
- PyTorch
- NumPy
- MoviePy (for video recording)
- tqdm (for progress visualization)


