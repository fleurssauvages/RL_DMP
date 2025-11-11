#usr/bin/env python3
"""
Multi-agent extension of PoWER-RL.
Maintains a population of PowerRL agents with mutual diversity pressure.
Each agent explores a different region of parameter space.
"""

import numpy as np
from scripts.power_rl import PowerRL

class MultiAgentPowerRL:
    def __init__(self, init_params, exploration_std, n_agents=5, reuse_top_n=6, diversity_strength=0.1):
        """
        init_params: (P,) np.array, initial parameter vector
        exploration_std: scalar or (P,), base exploration std dev
        n_agents: number of parallel agents
        diversity_strength: repulsion coefficient
        """
        self.n_agents = n_agents
        self.diversity_strength = diversity_strength
        self.exploration_std = exploration_std
        self.agents = []

        for i in range(n_agents):
            # small random perturbation in each agent's initialization
            theta_i = init_params + np.random.randn(*init_params.shape) * 0.1 * np.mean(exploration_std)
            agent_i = PowerRL(theta_i,
                              exploration_std=exploration_std,
                              reuse_top_n=reuse_top_n)
            self.agents.append(agent_i)

    def sample_policies(self):
        """Return list of sampled policies from each agent"""
        params = [agent.sample_policy() for agent in self.agents]
        return params

    def update_agents(self):
        """Update each agent individually using PoWER"""
        for agent in self.agents:
            agent.update()

    def apply_diversity_pressure(self, exploration_std=None, iteration=None, decay=0.98):
        if exploration_std is None:
            exploration_std = self.exploration_std
        """
        Apply repulsive forces between agents in parameter space, adaptively scaled.
        """
        thetas = np.array([a.theta for a in self.agents])
        n = len(self.agents)
        repulsion_range = 5.0 * np.mean(exploration_std)
        adaptive_strength = self.diversity_strength

        # Optional decay over iterations
        if iteration is not None:
            adaptive_strength *= decay ** iteration

        for i, agent in enumerate(self.agents):
            repulsion = np.zeros_like(agent.theta)
            for j in range(n):
                if i == j:
                    continue
                diff = agent.theta - thetas[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    dist = 0.001  # avoid division by zero
                # Exponentially damped repulsion, capped by distance range
                scale = np.exp(-dist / repulsion_range) / dist
                repulsion += diff * scale
            agent.theta += adaptive_strength * repulsion

    def best_agent(self):
        """Return index and reference to the best agent (by max recent return)"""
        best_R = -np.inf
        best_idx = 0
        for i, a in enumerate(self.agents):
            if len(a.history_returns) > 0 and np.max(a.history_returns) > best_R:
                best_R = np.max(a.history_returns)
                best_idx = i
        return best_idx, self.agents[best_idx]

    def reset_histories(self):
        """Clear rollout history of all agents"""
        for a in self.agents:
            a.reset_history()

    def update_exploration(self, new_std):
        for a in self.agents:
            a.update_exploration(new_std)
            
    def update_diversity_strength(self, new_strength):
        """
        Update (or decay) the repulsion/diversity strength between agents.
        Can be scalar or callable for adaptive control.
        """
        if np.isscalar(new_strength):
            self.diversity_strength = float(new_strength)
        else:
            raise ValueError("Diversity strength must be a scalar value.")