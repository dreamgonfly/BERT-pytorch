from torch.optim import Adam


class NoamOptimizer(Adam):

    def __init__(self, params, d_model, factor=2, warmup_steps=4000, betas=(0.9, 0.98), weight_decay=0, eps=1e-9):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor

        super(NoamOptimizer, self).__init__(params, betas=betas, weight_decay=weight_decay, eps=eps)

    def step(self, closure=None):
        self.step_num += 1
        self.lr = self._learning_rate()
        for group in self.param_groups:
            group['lr'] = self.lr
        super(NoamOptimizer, self).step()

    def _learning_rate(self):
        return self.factor * self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))