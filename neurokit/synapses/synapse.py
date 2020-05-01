class Synapse:
    def __init__(self, src, dest, context, w=0, d=1):
        self.src = src
        self.dest = dest
        self.context = context
        self.w = w
        self.d = d

        # d must be greater than 0
        if self.d <= 0:
            self.d = 1

    def register_spike(self):
        self.dest.register_potential_change(self.w, self.context.t() + self.d * self.context.dt())
