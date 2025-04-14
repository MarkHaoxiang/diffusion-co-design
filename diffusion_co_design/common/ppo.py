from diffusion_co_design.common.pydra import Config


class PPOConfig(Config):
    n_iters: int  # Number of training iterations
    n_epochs: int  # Number of optimization steps per training iteration
    minibatch_size: int  # Size of the mini-batches in each optimization step
    n_mini_batches: int  # Number of mini-batches in each epoch
    clip_epsilon: float  # clip value for PPO loss
    gamma: float  # discount factor
    lmbda: float  # lambda for generalised advantage estimation
    lr: float  # Learning rate
    max_grad_norm: float  # Maximum norm for the gradients
    entropy_eps: float  # coefficient of the entropy term in the PPO loss
    normalise_advantage: bool  # Whether to normalise the advantage estimates

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters
