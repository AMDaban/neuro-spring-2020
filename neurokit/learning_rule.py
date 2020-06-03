class Rule:
    pass


class STDP(Rule):
    def __init__(self, a_p, a_n, tau_p, tau_n):
        self.a_p = float(a_p)
        self.a_n = float(a_n)
        self.tau_p = float(tau_p)
        self.tau_n = float(tau_n)


class RMSTDP(Rule):
    pass
