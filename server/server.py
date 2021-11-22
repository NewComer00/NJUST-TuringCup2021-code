import os
import sys
import time
import numpy as np

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# self-defined modules to be added th PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(project_root, 'utils/')
sys.path.append(utils_path)
from turing_env import TuringEnv


PORT = 11111  # Port to listen on (non-privileged ports are > 1023)

game_path = r"D:\Desktop\workspace\csharp\turing2021\train-env\client\Build\TuringGame2021.exe"
env = TuringEnv(game_path=game_path, port=PORT)
state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n

MAP_LABELS = np.array([-2, -1, 0, 1, 2])
MOBILENET_IMG_SIZE = [96, 96]
FIRE_ANGLE_MAX = 360
FIRE_RANGE_MAX = 10

print("Training Environment Activated!")
print("========================================")

# ==================== PPO Utils ====================

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def ppo_loss(y_true, y_pred, oldpolicy_probs, advantages, rewards, values):
    newpolicy_probs = y_pred
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(rewards - values))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
        -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))

    return total_loss


# actor
def get_model_actor(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    n_actions = output_dims
    feature_extractor = MobileNetV2(
        input_shape=(*MOBILENET_IMG_SIZE, 3),
        weights='imagenet', include_top=False)
    for layer in feature_extractor.layers:
        layer.trainable = False
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='sigmoid')(x)
    model_actor = Model(
        inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
        outputs=[out_actions])
    model_actor.add_loss(ppo_loss(
        y_true=None,
        y_pred=out_actions,
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values))
    model_actor.compile(optimizer=Adam(learning_rate=1e-4))
    return model_actor


# critic
def get_model_critic(input_dims):
    state_input = Input(shape=input_dims)
    feature_extractor = MobileNetV2(
        input_shape=(*MOBILENET_IMG_SIZE, 3),
        weights='imagenet', include_top=False)
    for layer in feature_extractor.layers:
        layer.trainable = False
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_criticism = Dense(1, activation='linear')(x)
    model_critic = Model(inputs=[state_input], outputs=[out_criticism])
    model_critic.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model_critic


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        # map value to 0 ~ 255
        img_map = state.copy()
        img_norm = (img_map - MAP_LABELS.min()) / (MAP_LABELS.ptp() + np.finfo(float).eps)
        img_norm = np.around(img_norm * 255)
        img_color = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
        state = tf.image.resize(img_color, MOBILENET_IMG_SIZE)

        state_input = K.expand_dims(state, 0)
        action = model_actor.predict(
            [np.array([state]), np.array(dummy_n), np.array(dummy_1), np.array(dummy_1), np.array(dummy_1)],
            steps=1)
        action = action.squeeze()

        action_move = [np.around(action[0] * state_dims[0]),
                       np.around(action[1] * state_dims[1])]
        action_fire = [np.around(action[2] * state_dims[0]),
                       np.around(action[3] * state_dims[1])]
        next_state, reward, done, _ = env.step([*action_move, *action_fire])
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward


# ==================== PPO Global Init ====================

ppo_steps = 128
target_reached = False
best_reward = 0
iters = 0
max_iters = 50

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))
tensor_board = TensorBoard(log_dir='./server.log')

ACTOR_WEIGHT_PATH = "./model/actor_weight"
CRITIC_WEIGHT_PATH = "./model/critic_weight"

model_critic = get_model_critic(
    input_dims=(*MOBILENET_IMG_SIZE, 3))
model_actor = get_model_actor(
    input_dims=(*MOBILENET_IMG_SIZE, 3),
    output_dims=n_actions)
try:
    model_actor.load_weights(ACTOR_WEIGHT_PATH)
    model_critic.load_weights(CRITIC_WEIGHT_PATH)
except:
    print("No weight files found!")

while not target_reached and iters < max_iters:
    states = []
    actions = []
    values = []
    masks = []
    rewards = []

    for itr in range(ppo_steps):

        # map value to 0 ~ 255
        img_map = state.copy()
        img_norm = (img_map - MAP_LABELS.min()) / (MAP_LABELS.ptp() + np.finfo(float).eps)
        img_norm = np.around(img_norm * 255)
        img_color = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
        state = tf.image.resize(img_color, MOBILENET_IMG_SIZE)

        state_input = K.expand_dims(state, 0)
        action = model_actor.predict(
            [np.array([state]), np.array(dummy_n), np.array(dummy_1), np.array(dummy_1), np.array(dummy_1)],
            steps=1)
        action = action.squeeze()
        print(action)
        q_value = model_critic.predict(np.array([state]), steps=1)
        q_value = q_value.item()
        print(q_value)

        action_move = [np.around(action[0] * state_dims[0]),
                       np.around(action[1] * state_dims[1])]
        action_fire = [np.around(action[2] * state_dims[0]),
                       np.around(action[3] * state_dims[1])]
        observation, reward, done, info = env.step([*action_move, *action_fire])
        mask = not done

        states.append(state)
        actions.append(action)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)

        state = observation
        if done:
            break
            # env.reset()

    # map value to 0 ~ 255
    img_map = state.copy()
    img_norm = (img_map - MAP_LABELS.min()) / (MAP_LABELS.ptp() + np.finfo(float).eps)
    img_norm = np.around(img_norm * 255)
    img_color = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
    state = tf.image.resize(img_color, MOBILENET_IMG_SIZE)

    q_value = model_critic.predict(np.array([state]), steps=1)
    q_value = q_value.item()
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    actor_loss = model_actor.fit(
        [np.array(states),
         np.reshape(actions, newshape=(-1, 1, n_actions)),
         np.reshape(advantages, newshape=(-1, 1, 1)),
         np.reshape(rewards, newshape=(-1, 1, 1)),
         np.reshape(values[:-1], newshape=(-1, 1, 1))],
        [np.reshape(actions, newshape=(-1, n_actions))],
        verbose=True, shuffle=True, epochs=8,
        callbacks=[tensor_board])
    critic_loss = model_critic.fit(
        [np.array(states)],
        [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
        verbose=True, callbacks=[tensor_board])

    avg_reward = np.mean([test_reward() for _ in range(5)])
    print('total test reward=' + str(avg_reward))
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        model_actor.save_weights(ACTOR_WEIGHT_PATH)
        model_critic.save_weights(CRITIC_WEIGHT_PATH)
        best_reward = avg_reward
    if best_reward > 0.9 or iters > max_iters:
        target_reached = True
    iters += 1
    state = env.reset()

env.close()
