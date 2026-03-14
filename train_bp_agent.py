"""BP Agent Training Script"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import compute_gae, ppo_loss, collect_rollout, DEVICE
from utils.bp_dataset import BPDataset
from utils.raw_data import NUM_HEROES

LR = 3e-4


def train(epochs=32, batch_size=16, rollout_steps=5):
    # Load oracle
    oracle = WinRateOracle(embed_dim=128, nhead=8, num_layers=6, use_text=True, use_player_heroes=True).to(DEVICE)
    oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"
    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
        print(f"[+] Loaded oracle from {oracle_path}")
    oracle.eval()

    # Dataset: 优先使用合成数据，可选加载真实数据
    dataset = BPDataset(data_file="", num_synthetic=8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Agent
    agent = BPTransformerAgent(embed_dim=EMBED_DIM, nhead=8, num_layers=4).to(DEVICE)
    optimizer = AdamW(agent.parameters(), lr=LR)

    # Training loop
    for epoch in range(epochs):
        agent.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=90)
        for batch in pbar:
            # batch is a dict with keys: r_players, d_players, etc.
            # Convert to list of samples for rollout collection
            samples = []
            # Handle case where DataLoader returns list structure
            r_players_batch = batch['r_players']
            d_players_batch = batch['d_players']
            
            # Transpose if needed: from [5][160][batch_size] to [batch_size][5][160]
            if isinstance(r_players_batch, list) and len(r_players_batch) == 5:
                # DataLoader collates list of lists as [outer][inner][batch]
                batch_size = len(r_players_batch[0][0])
                r_players_batch = [[[r_players_batch[j][k][i] for k in range(len(r_players_batch[0]))] for j in range(5)] for i in range(batch_size)]
                d_players_batch = [[[d_players_batch[j][k][i] for k in range(len(d_players_batch[0]))] for j in range(5)] for i in range(batch_size)]
            else:
                batch_size = len(r_players_batch)
            
            for i in range(batch_size):
                sample = {
                    'r_players': r_players_batch[i],
                    'd_players': d_players_batch[i],
                }
                samples.append(sample)

            rollouts = [collect_rollout(agent, oracle, s) for s in samples[:rollout_steps]]

            for rollout in rollouts:
                actions = rollout['actions'].to(DEVICE)
                old_log_probs = rollout['log_probs'].to(DEVICE)
                values = rollout['values'].to(DEVICE)
                rewards = rollout['rewards'].to(DEVICE)

                T = len(rewards) - 1
                dones = torch.zeros(T, device=DEVICE)
                advantages, returns = compute_gae(rewards[:-1].unsqueeze(-1), values.unsqueeze(-1), dones.unsqueeze(-1))
                advantages = advantages.squeeze(-1)

                new_log_probs, new_values = [], []
                for i, state in enumerate(rollout['states']):
                    logits, v = agent(state)
                    heroes = state['action_history']['heroes']
                    used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    for h in range(1, NUM_HEROES + 1):
                        if (h - 1) not in used:
                            mask[h - 1] = 0.0
                    logits = logits + mask
                    probs = torch.softmax(logits, dim=-1)
                    new_log_probs.append(torch.distributions.Categorical(probs).log_prob(actions[i]))
                    new_values.append(v)

                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.cat(new_values)

                policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)
                value_loss = torch.nn.functional.mse_loss(new_values, returns)
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            pbar.set_postfix({"Loss": f"{total_loss / (len(pbar) * rollout_steps):.4f}"})

    # Save
    os.makedirs("./ckpts/bp_agent", exist_ok=True)
    torch.save(agent.state_dict(), f"./ckpts/bp_agent/bp_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
    print("[+] Training done!")


if __name__ == "__main__":
    train(epochs=32, batch_size=32, rollout_steps=10)
