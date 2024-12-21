import torch.nn as nn


class EnvCritic(nn.Module):
    supports_distillation: bool = False

    def predict_theta_value_with_hint(self, *args, **kwargs):
        if self.supports_distillation:
            return self.forward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs), None

    def predict_theta_value(self, *args, **kwargs):
        return self.predict_theta_value_with_hint(*args, **kwargs)[0]
