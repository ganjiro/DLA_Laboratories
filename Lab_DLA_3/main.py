import torch
from src.DQL import DQL

if __name__ == "__main__":
    train = False
    hidden_sizes = [128, 64]
    batch_size = 512
    eps_decay = 0.9999
    lr = 1e-3
    replay_memory_len = 150000
    num_episodes = 150000
    num_episodes_test = 1000

    path = 'checkpoints/last.pth'

    trainer = DQL(hidden_sizes, batch_size, eps_decay, path)

    if train:
        trainer.train(lr, replay_memory_len, num_episodes)
    else:
        trainer.q_function_1.load_state_dict(torch.load(path))

    trainer.test(num_episodes_test)
