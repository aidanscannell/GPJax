#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from gpjax.kernels import Rectangle, SquaredExponential
from gpjax.likelihoods import Bernoulli
from gpjax.mean_functions import Zero
from gpjax.models import SVGP
from jax.experimental import optimizers
from matplotlib import cm
from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

key = jax.random.PRNGKey(10)
cmap = cm.PiYG

input_dim = 2
output_dim = 1
num_inducing = 20


# Create training data
x1 = jnp.linspace(-2, 0, 30)
x2 = jnp.linspace(-8, 4, 30)
# X = jnp.stack([x1, x2], axis=-1)
X = jax.random.uniform(key=key, shape=(num_inducing, input_dim))
# X = jnp.array([-3.0, 1.0]).reshape(-1, input_dim)
Y = np.zeros([X.shape[0], 1])
Y[15:, 0] = 1
Y = jnp.array(Y)

# print(x.shape)

# xx, yy = jnp.meshgrid(states[:, 0], states[:, 1])
# states = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1)

# Create test inputs
x1_test = jnp.linspace(-4, 4, 10)
x2_test = jnp.linspace(-4, 4, 10)
x_test = jnp.stack([x1_test, x2_test], axis=-1)
xx_test, yy_test = jnp.meshgrid(x_test[:, 0], x_test[:, 1])
x_test = jnp.concatenate([xx_test.reshape(-1, 1), yy_test.reshape(-1, 1)], -1)

lengthscales = jnp.array([3.0, 1.0])
variance = 2
variance = 200
variance = 0.0005

# kernel = SquaredExponential(
#     lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
# )
kernel = Rectangle(lengthscales=lengthscales, variance=2.0)
mean_function = Zero(output_dim=output_dim)
likelihood = Bernoulli()

inducing_variable = jax.random.uniform(key=key, shape=(num_inducing, input_dim))
print("ind")
print(inducing_variable.shape)
print(X.shape)
svgp = SVGP(
    kernel, likelihood, inducing_variable, mean_function, num_latent_gps=output_dim
)
svgp_params = svgp.get_params()
print(svgp_params.keys())
print(svgp_params["q_sqrt"].shape)


print("x_test")
print(x_test.shape)
# Kdiag = gp_conditional_var(x, x_test, rectangle_cov_map)


def plot_model(params):
    fig, axs = plt.subplots(1, 2)
    mean, var = svgp.predict_y(params, x_test, full_cov=False)
    # mean, var = svgp.predict_f(params, x_test, full_cov=False)
    cont = axs[0].contourf(xx_test, yy_test, mean.reshape(xx_test.shape), cmap=cmap)
    fig.colorbar(cont, shrink=0.5, aspect=5, ax=axs[0])
    cont = axs[1].contourf(xx_test, yy_test, var.reshape(xx_test.shape), cmap=cmap)
    fig.colorbar(cont, shrink=0.5, aspect=5, ax=axs[1])
    axs[1].scatter(X[:, 0], X[:, 1])
    plt.show()


# plot_model(svgp_params)

learning_rate = 1e-3
batch = (X, Y)
num_epochs = 10000
num_epochs = 30000
num_epochs = 1

# Create optimizer
opt_init, opt_update, get_params = optimizers.adam(learning_rate)


# @jax.jit
def compute_loss(params, batch):
    X, Y = batch
    return -svgp.elbo(params, X, Y)


# @jax.jit
def train_step(step_i, opt_state, batch):
    params = get_params(opt_state)
    loss = compute_loss(params, batch)
    grads = jax.grad(compute_loss, argnums=0)(params, batch)
    return loss, opt_update(step_i, grads, opt_state)


opt_state = opt_init(svgp_params)

loss_history = []
for epoch in range(num_epochs):
    loss, opt_state = train_step(epoch, opt_state, batch)
    loss_history.append(loss.item())

    if epoch % 200 == 0:
        print("Loss @ epoch {} is {}".format(epoch, loss))

params = get_params(opt_state)
plot_model(params)
