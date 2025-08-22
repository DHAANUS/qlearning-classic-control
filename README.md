# Tabular Q-Learning — Classic Control (FrozenLake / CliffWalking)

From-scratch **tabular Q-Learning** with ε-greedy exploration. Includes
learning-curve plots and policy visualizations (heatmaps/arrows).

## Key ideas
- Update: \( Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a')\right] \)
- Off-policy TD control with replay-free, on-line updates
- ε-greedy with linear/exponential decay

## Quickstart
```bash
pip install gymnasium numpy matplotlib
# Open the notebook and run all cells:
# Q-Learning.ipynb
