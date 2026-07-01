import torch
import torch.nn as nn


# this focuses purely on RL mechanics

# The key idea that makes it "Group Relative" (instead of plain REINFORCE
# or PPO with a learned value function): for each prompt, sample a GROUP of
# G completions, then normalize each reward by that group's own mean/std.
# This replaces the critic/value-network PPO needs with simple statistics.

class TinyPolicy(nn.Module):
    """Given two digits (a, b), outputs a distribution over 19 possible sums."""

    def __init__(self, hidden = 64):
        super().__init__()
        self.a_emb = nn.Embedding(10, hidden)  # shape: (B, ) -> (B, hidden_dim)
        self.b_emb = nn.Embedding(10, hidden)
    
        self.net = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 19)
        )

    def forward(self, a, b):
        x = torch.cat([self.a_emb(a), self.b_emb(b)], dim = -1)

        # shape: after emb: shape: (B, ) -> (B, hidden_dim)
        # shape after torch.cat(a_emb, b_emb) -> (B, 2* hidden_dim)
        return self.net(x) # (B, 19)
        

if __name__ == "__main__":
    batch_size = 4
    a = torch.randint(0, 10, (batch_size,))
    b = torch.randint(0, 10, (batch_size,))

    print(a.shape, b.shape)
    true_answer = a + b

    policy = TinyPolicy()
    out_pred = policy(a,b)

    print(f"True ans: {true_answer}, predicted_output: {out_pred.shape}")

