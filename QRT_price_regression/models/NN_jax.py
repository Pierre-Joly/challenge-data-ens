import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import linen as nn  # Flax is a neural network library for JAX
from flax.training import train_state  # Helper for managing training state
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import optax

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

@jax.jit
def train_step(state, X, Y):
    def loss_fn(params):
        predictions = Net().apply({'params': params}, X)
        loss = jnp.mean((predictions - Y) ** 2)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def metric(output, y):
    return spearmanr(output, y).correlation

def preprocess(X, Y=None):
    X['COUNTRY'] = X['COUNTRY'].apply(lambda x: -1 if x == 'FR' else (1 if x == 'DE' else 0))
    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    if Y is not None:
        Y = Y['TARGET']
        return X, Y
    return X

X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')

X_train, Y_train = preprocess(X_train, Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
input_features = len(X_train.columns)

# Initialize model and optimizer
model = Net()
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, input_features]))['params']
optimizer = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer(0.001))

# Training loop
for epoch in range(100):  # Number of epochs
    state, loss = train_step(state, X_train, Y_train)
    print(f'Epoch {epoch+1}, Loss: {loss}')

