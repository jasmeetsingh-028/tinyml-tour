import torch
import torch.nn as nn
from torch.distributions import Categorical

def step(model, optimizer, batch_size = 64, group_size = 8):

    # repeat each problem G times 
    # this is "the group"
    # a, b, and true_answer shape: (B, )

    a = torch.randint(0, 10, (batch_size,))
    b = torch.randint(0, 10, (batch_size,))

    true_answer = a + b

    a_rep = a.repeat_interleave(group_size) #shape: (B * group_size, )
    b_rep = b.repeat_interleave(group_size)

    true_rep = true_answer.repeat_interleave(group_size)  # shape: (B * group_size, )

    logits = model(a_rep, b_rep) # shape: (B * group_size, 19)

    dist = Categorical(logits=logits)

    actions = dist.sample()              # the model's G attempts per problem

    log_probs = dist.log_prob(actions)

    reward = (actions == true_rep).float()   # rule-based reward, just like real GRPO

    # --- THE core GRPO step: normalize reward within each group ---

    reward_grouped = reward.view(batch_size, group_size)

    mean = reward_grouped.mean(dim=1, keepdim=True)
    std = reward_grouped.std(dim=1, keepdim=True) + 1e-4
    advantage = ((reward_grouped - mean) / std).view(-1)

    # policy gradient: push up log-prob of actions with above-average reward
    # in their group, push down log-prob of below-average ones.

    loss = -(advantage.detach() * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return reward.mean().item() # return the average reaward across the batch



