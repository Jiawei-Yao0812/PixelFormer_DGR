import torch
import torch.nn.functional as F
import numpy as np
import ot

class OptimalTransportDepthLoss(torch.nn.Module):
    def __init__(self, lambda_otdl=1.0):
        super(OptimalTransportDepthLoss, self).__init__()
        self.lambda_otdl = lambda_otdl

    def forward(self, P, Q):
        # Normalize depth maps
        P_prime = P / torch.sum(P, dim=[2, 3], keepdim=True)
        Q_prime = Q / torch.sum(Q, dim=[2, 3], keepdim=True)

        # Compute cost matrix M
        depth_values = torch.arange(P.size(2)).to(P.device)
        M = torch.abs(depth_values.view(-1, 1) - depth_values.view(1, -1))**2
        M = M.detach().cpu().numpy()

        # Optimal Transport calculation
        batch_size = P.size(0)
        OT_values = []
        for b in range(batch_size):
            P_np = P_prime[b].detach().cpu().numpy()
            Q_np = Q_prime[b].detach().cpu().numpy()
            OT_value = ot.emd2(P_np.ravel(), Q_np.ravel(), M)  # Earth Mover's Distance
            OT_values.append(OT_value)
        OT_values = torch.tensor(OT_values, device=P.device)

        # MSE loss
        mse_loss = F.mse_loss(P, Q, reduction='none').mean(dim=[1, 2, 3])

        # Final combined loss
        loss = mse_loss + self.lambda_otdl * OT_values

        return loss.mean()
