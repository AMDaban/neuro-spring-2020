class NeuronState:
    def __init__(self, u, t):
        self.u = u
        self.t = t

    def __str__(self):
        return f"u: {self.u}, t: {self.t}"
