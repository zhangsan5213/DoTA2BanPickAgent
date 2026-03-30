# Dota 2 Ban Pick Agent

A Reinforcement Learning project that trains an AI agent to perform intelligent drafting (ban/pick) in Dota 2 Captain Mode (CM). The agent learns to make optimal decisions that maximize win probability by taking into account player hero preferences, hero synergies, and counterplay.

## Key Features

- **Player-aware drafting**: Incorporates individual player win rates on heroes, unlike generic drafting agents
- **Multi-modal hero encoding**: Combines structured hero attributes with text embeddings from ability descriptions
- **Two-stage training**: Supervised pre-trained WinRateOracle provides reward signal for RL, eliminating need for full game playouts
- **Continuous improvement**: Uses ELO/TrueSkill rating systems to evaluate agent versions against historical checkpoints
- **Follows official competitive rules**: Implements the exact Captain Mode ban/pick sequence
- **Modern RL**: Uses Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)

## How It Works

### Drafting Order (Official CM Mode)
The environment follows the standard competitive Dota 2 Captain Mode draft sequence (20 total steps):
1. **Ban Phase 1**: Radiant → Dire → Radiant → Dire (4 bans)
2. **Pick Phase 1**: Radiant → Dire → Dire → Radiant (4 picks)
3. **Ban Phase 2**: Dire → Radiant → Dire → Radiant (4 bans)  
4. **Pick Phase 2**: Dire → Radiant → Radiant → Dire (4 picks)
5. **Ban Phase 3**: Radiant → Dire (2 bans)
6. **Pick Phase 3**: Radiant → Dire (2 picks)

Total: **10 bans (5 per team) + 10 picks (5 per team)**

### Architecture

1. **WinRateOracle**: Pre-trained neural network that predicts win probability given final team compositions and player preferences. Provides the reward signal for RL training. Currently achieves **~90.4% prediction accuracy** on held-out high MMR match data.

2. **BPTransformerAgent**: Transformer-based policy/value network that processes:
   - **Player preferences**: Encodes each player's historical win rates on heroes
   - **Action history**: Encodes previous bans/picks
   - Produces policy logits for next action and state value estimate

3. **Environment**: Implements the full CM draft process with proper masking of unavailable heroes.

## Project Structure

```
├── configs/              # Configuration files
│   └── bp_agent_config.yaml
├── data/                 # Data files
│   ├── hero_features.xlsx
│   ├── hero_semantic_embeddings.pt
│   ├── hero_ability_descriptions.json
│   ├── hero_positions.json
│   └── high_mmr_with_stats.json
├── model/                # Neural network models
│   ├── bp_agent.py       # Main BP Transformer Agent
│   ├── win_rate_oracle.py# Win rate prediction oracle
│   └── hero_encoder.py   # Multi-modal hero encoder
├── utils/                # Utilities
│   ├── bp_env.py         # RL environment implementation
│   ├── raw_data.py       # Data loading utilities
│   └── player_preference_sampler_optimized.py  # Data generation
├── eval/                 # Evaluation and rating systems
│   ├── elo_rating.py     # ELO rating implementation
│   └── trueskill_rating.py  # TrueSkill rating implementation
├── trainer/              # Modular training components (WIP)
├── ckpts/                # Model checkpoints
├── runs/                 # TensorBoard logs
├── train_winrate_oracle.py   # Script to train the oracle
├── train_bp_agent.py         # Main RL training script
└── eval_bp_agent.py          # Evaluation tournament script
```

## Technologies

- **Framework**: PyTorch + Transformers
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) with GAE
- **Data Processing**: Pandas, NumPy
- **Rating**: ELO, TrueSkill
- **Logging**: TensorBoard

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/DoTA2BanPickAgent.git
cd DoTA2BanPickAgent

# Install dependencies
pip install torch pandas numpy openpyxl tqdm pyyaml tensorboard trueskill
```

## Usage

### 1. Train the WinRateOracle (or use pre-trained)

A pre-trained checkpoint is already included in `./ckpts/`. To train from scratch:

```bash
python train_winrate_oracle.py
```

### 2. Train the BP Agent

```bash
python train_bp_agent.py
```

Training uses the configuration from `configs/bp_agent_config.yaml`. Key settings:

- `rating.method`: `"elo"` or `"trueskill"` (default: trueskill)
- `training.epochs`: Number of training epochs
- `training.historical_opponent_prob`: Probability of sampling historical opponents (default 0.6 = 60%)
- `actor_lr`: Learning rate for the policy network

### 3. Evaluate and watch agents battle

Run a round-robin tournament between top-rated models:

```bash
# Watch top 3 models play 3 matches each
python eval_bp_agent.py --top_n 3 --matches 3

# Evaluate specific model checkpoints
python eval_bp_agent.py --models ./ckpts/model1.pth ./ckpts/model2.pth --matches 5

# Use ELO for selecting top models
python eval_bp_agent.py --top_n 3 --rating elo
```

The evaluation script will print step-by-step drafts with hero names and output final standings with win rates.

## Performance

- **WinRateOracle**: Achieves 0.9042 validation accuracy predicting match outcomes from team compositions and player preferences
- **BP Agent**: Gradually improves through self-play and historical opponent evaluation, with the TrueSkill rating system naturally selecting the best agents

## Training Approach

1. **Supervised Pre-training**: The WinRateOracle is trained on real high MMR match data to predict win probabilities
2. **RL Fine-tuning**: The agent plays against itself and historical versions, with the oracle providing reward at the end of each draft
3. **Continuous Evaluation**: Every N epochs, new agents battle existing checkpoints and get rated
4. **Selection**: Higher-rated agents are more likely to be sampled as opponents for future training, driving progressive improvement

## License

MIT