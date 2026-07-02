"""
Federated Learning stub for SupplyMind.

Simulates 3 companies training on private data, sharing only model
parameters (not raw data) via FedAvg.

Hypothesis under test: a federated model may generalize better than any single
company's model because each company observes different disruptions in its
supply-chain segment. Whether that holds — and by how much — is only whatever
a real run measures; no result is claimed here.

Usage:
    python -m rl.federated.fedavg
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class FederatedSupplyMindTrainer:
    """FedAvg training across multiple simulated companies.

    Splits the offline buffer into n_clients private datasets.
    Each client trains locally for local_epochs, then shares only
    model parameters. Global model = average of client models.
    Optional: differential privacy noise on shared parameters.

    Args:
        n_clients:     Number of federated clients (3 = Apple, Samsung, Toyota).
        n_rounds:      Number of federation rounds.
        local_epochs:  Local training epochs per round.
        dp_noise_std:  Gaussian noise std for differential privacy (0 = no DP).
    """

    def __init__(
        self,
        n_clients: int = 3,
        n_rounds: int = 20,
        local_epochs: int = 5,
        dp_noise_std: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.dp_noise_std = dp_noise_std
        self.device = device

        self.client_names = ["Apple", "Samsung", "Toyota"][:n_clients]

    def _create_model(self) -> nn.Module:
        """Create a fresh BC-style model for each client."""
        from rl.offline.baselines import BCNetwork
        return BCNetwork(408, 280)

    def _split_data(
        self, states: np.ndarray, actions: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split dataset into n_clients private subsets."""
        n = len(states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.n_clients)
        return [(states[s], actions[s]) for s in splits]

    def _fedavg(self, models: list[nn.Module]) -> dict[str, torch.Tensor]:
        """Average model parameters across clients (FedAvg)."""
        avg_state = {}
        for key in models[0].state_dict():
            tensors = [m.state_dict()[key].float() for m in models]
            avg_state[key] = torch.stack(tensors).mean(dim=0)
        return avg_state

    def _add_dp_noise(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add Gaussian noise for differential privacy."""
        if self.dp_noise_std <= 0:
            return state_dict
        noisy = {}
        for key, tensor in state_dict.items():
            noise = torch.randn_like(tensor) * self.dp_noise_std
            noisy[key] = tensor + noise
        return noisy

    def train(self) -> dict[str, Any]:
        """Run federated training.

        Returns dict with per-round metrics and final model.
        """
        DATA_DIR = Path(__file__).resolve().parent.parent / "data"
        npz_path = DATA_DIR / "offline_buffer.npz"

        if not npz_path.exists():
            logger.warning("No offline buffer. Using random data for demo.")
            states = np.random.randn(1000, 408).astype(np.float32)
            actions = np.random.randint(0, 280, size=1000).astype(np.int64)
        else:
            data = np.load(str(npz_path))
            states = data["states"]
            raw_actions = data["actions"]
            actions = (raw_actions[:, 0] * 40 + raw_actions[:, 1]).astype(np.int64)

        # Split data
        client_data = self._split_data(states, actions)
        logger.info("Federated training: %d clients, %d rounds, %d local epochs",
                     self.n_clients, self.n_rounds, self.local_epochs)
        for i, (s, a) in enumerate(client_data):
            logger.info("  Client %s: %d transitions", self.client_names[i], len(s))

        # Initialize global model
        global_model = self._create_model()
        criterion = nn.CrossEntropyLoss()

        round_metrics = []

        for round_idx in range(self.n_rounds):
            client_models = []

            for client_idx in range(self.n_clients):
                # Clone global model for this client
                local_model = copy.deepcopy(global_model).to(self.device)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)

                s_data, a_data = client_data[client_idx]
                s_tensor = torch.from_numpy(s_data).to(self.device)
                a_tensor = torch.from_numpy(a_data).to(self.device)

                # Local training
                local_model.train()
                for _ in range(self.local_epochs):
                    # Mini-batch
                    perm = torch.randperm(len(s_tensor))[:512]
                    logits = local_model(s_tensor[perm])
                    loss = criterion(logits, a_tensor[perm])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                client_models.append(local_model.cpu())

            # FedAvg + optional DP noise
            avg_params = self._fedavg(client_models)
            if self.dp_noise_std > 0:
                avg_params = self._add_dp_noise(avg_params)
            global_model.load_state_dict(avg_params)

            # Evaluate global model
            global_model.eval()
            with torch.no_grad():
                all_s = torch.from_numpy(states[:1000])
                all_a = torch.from_numpy(actions[:1000])
                logits = global_model(all_s)
                acc = (logits.argmax(dim=-1) == all_a).float().mean().item()

            round_metrics.append({"round": round_idx + 1, "global_accuracy": round(acc, 4)})

            if (round_idx + 1) % 5 == 0:
                logger.info("  Round %d/%d: global accuracy=%.4f",
                            round_idx + 1, self.n_rounds, acc)

        # Save
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHECKPOINT_DIR / "federated_global.pt"
        torch.save({"state_dict": global_model.state_dict(), "rounds": self.n_rounds}, str(save_path))

        return {
            "rounds": round_metrics,
            "final_accuracy": round_metrics[-1]["global_accuracy"],
            "n_clients": self.n_clients,
            "dp_noise_std": self.dp_noise_std,
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    trainer = FederatedSupplyMindTrainer(n_clients=3, n_rounds=20, local_epochs=5, device="cpu")
    result = trainer.train()
    print(f"Final federated accuracy: {result['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
