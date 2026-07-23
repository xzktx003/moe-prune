import torch

def gram_linear(x):
    """Compute the Gram matrix using a linear kernel."""
    return x @ x.T

def gram_rbf(x, sigma=None):
    """Compute the Gram matrix using an RBF kernel."""
    if sigma is None:
        sigma = x.shape[1]  # Default sigma is the feature dimension
    norms = torch.sum(x ** 2, dim=1, keepdim=True)
    dists = norms + norms.T - 2 * x @ x.T
    return torch.exp(-dists / (2.0 * sigma ** 2))

def centered_gram(gram):
    """Center the Gram matrix."""
    n = gram.size(0)
    ones = torch.ones(n, n, device=gram.device) / n
    return gram - ones @ gram - gram @ ones + ones @ gram @ ones

def cka(gram_x, gram_y):
    """Compute CKA."""
    gram_x_centered = centered_gram(gram_x)
    gram_y_centered = centered_gram(gram_y)

    numerator = torch.sum(gram_x_centered * gram_y_centered)
    denominator = torch.sqrt(torch.sum(gram_x_centered ** 2) * torch.sum(gram_y_centered ** 2))

    return numerator / denominator

def cka_linear(x, y):
    """Compute linear CKA."""
    gram_x = gram_linear(x)
    gram_y = gram_linear(y)
    return cka(gram_x, gram_y)

def cka_rbf(x, y, sigma=None):
    """Compute RBF CKA."""
    gram_x = gram_rbf(x, sigma)
    gram_y = gram_rbf(y, sigma)
    return cka(gram_x, gram_y)

# Example usage:
x = torch.randn(100, 50)  # Example feature matrix for dataset X
y = torch.randn(100, 50)  # Example feature matrix for dataset Y

cka_value_linear = cka_linear(x, y)
cka_value_rbf = cka_rbf(x, y)

print("Linear CKA:", cka_value_linear.item())
print("RBF CKA:", cka_value_rbf.item())