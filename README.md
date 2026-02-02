# Gumbel AlphaZero for Chinese Chess (Xiangqi)

Implementation of Gumbel AlphaZero algorithm with Sequential Halving for Chinese Chess (Xiangqi). This system replaces traditional MCTS with Gumbel sampling and Sequential Halving for more efficient exploration.

## Overview

Gumbel AlphaZero is an improvement over standard AlphaZero that uses:
- **Gumbel Sampling**: Adds Gumbel noise to policy logits for exploration
- **Sequential Halving**: Progressively narrows action space without full tree search
- **Dual-Head Network**: ResNet backbone with policy and value heads
- **Offline Training**: Trains on 1.88 million game records from CSV data

## Project Structure

```
cchess-Gumbel-AlphaZero/
├── src/
│   ├── encoding/          # Data encoding (board, actions)
│   ├── network/           # Neural network (ResNet + heads)
│   ├── gumbel/            # Gumbel search engine
│   ├── training/          # Training pipeline
│   ├── utils/             # Utilities (logging, background tasks)
│   └── inference/         # Inference modules
├── scripts/               # Training and evaluation scripts
├── config/                # YAML configurations
├── data/                  # Training data
├── models/                # Trained models
└── logs/                  # Training logs
```

## Installation

```bash
# Clone repository
cd cchess-Gumbel-AlphaZero

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The training data is in CSV format:
```
chessboradStatus,moveStatus,chessboradStatusAfterMoving,predictAction,result
0919293949596979891777062646668600102030405060708012720323436383,7747,0919293949596979891747062646668600102030405060708012720323436383,7062,None
```

- **chessboradStatus**: 64-character board encoding
- **moveStatus**: 4-digit move (e.g., "7747" = from position 77 to 47)
- **result**: 0=Red wins, 1=Black wins, 2=Draw, None=incomplete

## Network Architecture

```
Input (14, 10, 9)  # 14 channels: 7 Red + 7 Black pieces
    ↓
Initial Conv (256 filters, 3×3)
    ↓
Residual Blocks × 15
    ↓
    ├─→ Policy Head → 2086 logits → softmax
    └─→ Value Head → 256 → tanh → [-1, 1]
```

## Training

### Phase 1: Supervised Pre-training

```bash
python scripts/train_supervised.py \
    --config config/training.yaml \
    --data data/chess.csv \
    --output models/final/
```

### Background Training

```bash
# Train in background
python scripts/train_supervised.py \
    --config config/training.yaml \
    --data data/chess.csv \
    --background \
    --task-id train_001

# Check status
python scripts/background_status.py --task-id train_001

# Wait for completion
python scripts/background_status.py --task-id train_001 --wait
```

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

## Gumbel Search

The search algorithm uses:
1. Get initial policy π from network
2. Sequential Halving: iteratively select top-k actions
3. Simulate each action to depth d
4. Compute improved policy π̂ from Q-values

```python
from src.network.alphanet import AlphaZeroNet
from src.gumbel.search import GumbelSearch

# Load network
network = AlphaZeroNet()
network.load_state_dict(torch.load('model.pth'))

# Create search engine
search = GumbelSearch(network)

# Run search
result = search.search(board_tensor, legal_moves, is_root=True)
best_action = search.get_best_action(result)
```

## Configuration

Edit config files in `config/`:
- `network.yaml`: Network architecture
- `training.yaml`: Training hyperparameters
- `gumbel.yaml`: Search parameters

## Success Metrics

**Phase 1 (Supervised):**
- Policy accuracy > 40% (expert move prediction)
- Value MSE < 0.3
- Training speed > 1000 positions/sec

**Phase 2 (Gumbel):**
- Win rate vs. supervised > 60%
- Search improvement > 0.2
- Training stability < 5% variance

## Logging

Logs are saved in `logs/`:
- `training.log`: Main training log
- `gumbel.log`: Search log
- `tensorboard/`: TensorBoard logs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## References

- [Gumbel AlphaZero Paper](https://arxiv.org/abs/2407.04678v1)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [cchess-Alpha-Beta+NNUE](../cchess-Alpha-Beta+NNUE) - Reference implementation

## License

MIT License
