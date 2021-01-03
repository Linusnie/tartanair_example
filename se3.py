import torch


class Se3:
    def __init__(self, r: torch.Tensor=None, t: torch.Tensor=None):
        if r is None and t is None:
            r = torch.eye(3)
            t = torch.zeros(3)
        elif r is None:
            r = torch.eye(3).repeat(*t.shape[:-1], 1, 1)

        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.get_default_dtype())

        if r.shape[-2:] == (3, 3):
            self.r = r
        elif r.shape[-1] == 4:
            self.r = q_to_r(r)
        else:
            raise ValueError(f'Rotation must have dimension (..., 3, 3) or (..., 4), got {r.shape}')
        self.shape = self.r.shape[:-2]

        if t is None:
            t = torch.zeros(*self.shape, 3)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        if t.shape[-1] != 3:
            raise ValueError(f'Translation must have dimension (..., 3), got {t.shape}')
        self.t = t

        if self.t.shape[:-1] != self.r.shape[:-2]:
            raise ValueError(f'Dimension mismatch, t: {t.shape}, r: {r.shape}')

    def inv(self):
        r_inv = self.r.transpose(-2, -1)
        return Se3(
            r=r_inv,
            t=-torch.matmul(r_inv, self.t.unsqueeze(-1))[..., 0]
        )

    def __mul__(self, other):
        return Se3(
            r=torch.matmul(self.r, other.r),
            t=torch.matmul(self.r, other.t.unsqueeze(-1))[..., 0] + self.t
        )

    def __getitem__(self, item):
        return Se3(self.r.__getitem__(item), self.t.__getitem__(item))

    def __repr__(self):
        return f'R: {self.r}, t: {self.t}'

    def __len__(self):
        return self.shape[0]

    def reshape(self, *indices):
        return Se3(self.r.reshape(*indices, 3, 3), t=self.t.reshape(*indices, 3))

    def transform_points(self, points):
        return torch.matmul(self.r, points) + self.t.unsqueeze(-1)


def q_to_r(q, real_first=False):
    if real_first:
        a, b, c, d = [q[..., i] for i in range(4)]
    else:
        b, c, d, a = [q[..., i] for i in range(4)]

    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    ab = 2 * a * b
    ac = 2 * a * c
    ad = 2 * a * d
    bc = 2 * b * c
    bd = 2 * b * d
    cd = 2 * c * d
    return torch.stack([
        torch.stack([a2 + b2 - c2 - d2, bc + ad, bd - ac], dim=-1),
        torch.stack([bc - ad, a2 - b2 + c2 - d2, cd + ab], dim=-1),
        torch.stack([bd + ac, cd - ab, a2 - b2 - c2 + d2], dim=-1)
    ], dim=-1)
