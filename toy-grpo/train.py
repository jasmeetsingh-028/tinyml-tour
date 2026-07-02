import torch
import torch.nn as nn

from model.policy import TinyPolicy
from grpo.grpo_step import step

def train(model, optmizer, num_steps = 1000):

    print("Toy GRPO on ""addition of two digits (0-9)")

    for steps in range(num_steps):
        acc = step(model, optmizer)
        if steps % 100 == 0:
            print(f"Step: {steps}, acc: {acc:.3f}")
    

     # final eval, greedy
    a = torch.arange(10).repeat_interleave(10)
    b = torch.arange(10).repeat_interleave(10)

    with torch.no_grad():
        pred = model(a,b).argmax(dim = -1)
    
    final_acc = (pred == (a + b)).float().mean().item()
    print(f"\nFinal greedy accuracy on all 100 (a,b) pairs: {final_acc:.1%}\n")


if __name__ == "__main__":
    model = TinyPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    train(model, optimizer, num_steps = 2000)