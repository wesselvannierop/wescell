import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class IICloss:
    """Loss function that returns marginal entropy and conditional using the predictions from a data pair.
    This can be used to maximize mutual information (MI) according to:
    mutual_information = marginal_entropy - conditional_entropy

    To maximize it using a minimizer: loss = - mutual_information.

    Parameters:
    consider_neighbouring_pixels (int): Nr. of neightbouring pixels to consider during optimization.

    Returns:
    marginal_entropy (float): Quantity that measures the amount of randomness in the prediction. Maximizing it will equilize classes.
    conditional_entropy (float): Quantity that measures the amount of information needed to describe the outcome of a random variable Y given that the value of another random variable Y'. Minimizing will force the random variables / predictions to be similar.
    """

    def __init__(
        self,
        consider_neighbouring_pixels: int = 1,
        entropy_coeff: float = 1.0,
        nr_of_clusters: int = 2,
    ) -> None:
        """
        Args:
            consider_neighbouring_pixels (int, optional): Nr. of neightbouring pixels to consider during optimization. Defaults to 1.
            entropy_coeff (float, optional): How much weight to add to maximizing marginal entropy (equal class predictions). Defaults to 1.0.
            nr_of_clusters (int, optional): Number of classes the model should try to identify. Defaults to 2.
        """
        self.consider_neighbouring_pixels = consider_neighbouring_pixels
        self.entropy_coeff = entropy_coeff
        self.nr_of_clusters = nr_of_clusters

    def conditional_entropy(
        self, P: torch.FloatTensor, P_marginal: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Computes conditional entropy.
            .. math::
                \mathrm {H} (Y|X)\ =-\sum _{x\in {\mathcal {X}},y\in {\mathcal {Y}}}p(x,y)\log {\frac {p(x,y)}{p(x)}}

        Args:
            P (torch.FloatTensor): Joint probability distribution matrix.
            P_marginal (torch.FloatTensor): Marginal probability distribution vector.

        Returns:
            torch.FloatTensor: Conditional entropy (single float)
        """
        return -(P * (torch.log(P) - torch.log(P_marginal.repeat(1, self.nr_of_clusters)))).sum()

    @staticmethod
    def marginal_entropy(P_marginal: torch.FloatTensor) -> torch.FloatTensor:
        """Computes entropy for a marginal distribution (vector).
            .. math::
                \mathrm {H} (X)\ =-\sum _{x\in {\mathcal {X}}} p(x)\log p(x)

        Args:
            P_marginal (torch.FloatTensor): Vector describing the marginal distribution.

        Returns:
            torch.FloatTensor: Marginal entropy (single float)
        """
        return -(P_marginal * torch.log(P_marginal)).sum()

    def joint_probability_distribution_matrix(
        self, pred1: torch.FloatTensor, pred2: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute the joint probability distribution matrix. This is a `nr_of_clusters` x `nr_of_clusters` matrix containing the probability of a certain class given another class.

        Args:
            pred1 (torch.FloatTensor): One of the predictions in the data pair.
            pred2 (torch.FloatTensor): One of the predictions in the data pair.

        Returns:
            torch.FloatTensor: Joint probability distribution matrix.
        """
        # Format for Convolution
        pred1 = pred1.permute(1, 0, 2, 3)
        pred2 = pred2.permute(1, 0, 2, 3)

        # Get joint_probability_distribution_matrix
        P = nn.functional.conv2d(
            pred1, weight=pred2, padding=self.consider_neighbouring_pixels
        )  # k, k, 2 * padding + 1,2 * padding + 1
        P = P.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

        # Normalise
        current_norm = float(P.sum())
        P = P / current_norm

        # Symmetrise
        return (P + P.t()) / 2.0

    def __call__(
        self, pred1: torch.FloatTensor, pred2: torch.FloatTensor
    ) -> list[torch.FloatTensor]:
        assert pred1.shape == pred2.shape

        # Crop 2 px of the edge to reduce artifacts of the random affine transformation
        center_crop = [item - 2 for item in pred1.shape[2:]]
        pred1 = TF.center_crop(pred1, center_crop)
        pred2 = TF.center_crop(pred2, center_crop)

        P = self.joint_probability_distribution_matrix(pred1, pred2)

        # Compute marginals
        Pc1 = P.sum(dim=1).unsqueeze(1)  # k, 1
        Pc2 = P.sum(dim=0).unsqueeze(0)  # 1, k

        # For log stability
        EPS = 1e-7
        P[(P < EPS).data] = EPS
        Pc1[(Pc1 < EPS).data] = EPS
        Pc2[(Pc2 < EPS).data] = EPS

        # Compute entropy and loss
        marginal_entropy = self.marginal_entropy(Pc1)
        conditional_entropy = self.conditional_entropy(P, Pc1)
        loss = -(
            marginal_entropy - conditional_entropy + 2 * (self.entropy_coeff - 1) * marginal_entropy
        )

        return loss, marginal_entropy, conditional_entropy
