import os
import socket
import json
import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 11111  # Port to listen on (non-privileged ports are > 1023)

RECV_BUF_MAX_SIZE = 10240  # must be larger than the size of a socket frame
FRAME_LEN_BYTES = 4
HEADER = b'\xaa\xaa'
EOF = b'\xa5\xa5'

ACTION_TYPE = ["MoveX", "MoveY", "FireX", "FireY"]

MAP_LABELS = np.array([-2, -1, 0, 1, 2])
MOBILENET_IMG_SIZE = [96, 96]

MAP_SHAPE = [50, 50]
FIRE_ANGLE_MAX = 360
FIRE_RANGE_MAX = 10

print("Python Server Activated!")
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


# ==================== PPO Global Init ====================

MAX_GAME_TURN = 10
current_step = 0
current_game_turn = 0
dummy_n = np.zeros((1, 1, len(ACTION_TYPE)))
dummy_1 = np.zeros((1, 1, 1))
tensor_board = TensorBoard(log_dir='./server.log')

ACTOR_WEIGHT_PATH = "./model/actor_weight"
CRITIC_WEIGHT_PATH = "./model/critic_weight"

model_critic = get_model_critic(
    input_dims=(*MOBILENET_IMG_SIZE, 3))
model_actor = get_model_actor(
    input_dims=(*MOBILENET_IMG_SIZE, 3),
    output_dims=len(ACTION_TYPE))
try:
    model_actor.load_weights(ACTOR_WEIGHT_PATH)
    model_critic.load_weights(CRITIC_WEIGHT_PATH)
except:
    print("No weight files found!")

# ==================== Socket Init ====================

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while current_game_turn < MAX_GAME_TURN:

            # ==================== Receive Game Data ====================

            recv_buf = conn.recv(RECV_BUF_MAX_SIZE)
            if len(recv_buf) > 0:
                recv_frame = recv_buf[recv_buf.find(HEADER):recv_buf.rfind(EOF) + len(EOF)]
                # get the frame length info stored in the frame
                recv_frame_len = int.from_bytes(
                    recv_frame[len(HEADER):len(HEADER) + FRAME_LEN_BYTES],
                    byteorder='big', signed=True)
                # if len(frame) is zero, it means current buffer doesn't contain a valid frame
                # if the frame length is correct and nonzero, get the payload from it
                if len(recv_frame) == recv_frame_len and len(recv_frame) > 0:
                    recv_payload = recv_frame[len(HEADER) + FRAME_LEN_BYTES:-len(EOF)]

                    # ==================== Append Data to PPO Lists ====================

                    if current_step == 0:
                        states = []
                        actions = []
                        values = []
                        masks = []
                        rewards = []

                    elif current_step > 0:
                        masks.append(mask)
                        rewards.append(reward)

                        # if game is over 
                        if not mask:
                            # ==================== Calc PPO Lost ====================

                            # print(len(states))
                            # print(len(actions))
                            # print(len(values))
                            # print(len(masks))
                            # print(len(rewards))

                            q_value = model_critic.predict(np.array([state]), steps=1)
                            q_value = q_value.item()
                            values.append(q_value)
                            returns, advantages = get_advantages(values, masks, rewards)
                            actor_loss = model_actor.fit(
                                [np.array(states),
                                 np.reshape(actions, newshape=(-1, 1, len(ACTION_TYPE))),
                                 np.reshape(advantages, newshape=(-1, 1, 1)),
                                 np.reshape(rewards, newshape=(-1, 1, 1)),
                                 np.reshape(values[:-1], newshape=(-1, 1, 1))],
                                [np.reshape(actions, newshape=(-1, len(ACTION_TYPE)))],
                                verbose=True, shuffle=True, epochs=8,
                                callbacks=[tensor_board])
                            critic_loss = model_critic.fit(
                                [np.array(states)],
                                [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                verbose=True, callbacks=[tensor_board])

                            model_actor.save_weights(ACTOR_WEIGHT_PATH)
                            model_critic.save_weights(CRITIC_WEIGHT_PATH)

                            # avg_reward = np.mean([test_reward() for _ in range(5)])
                            # print('total test reward=' + str(avg_reward))
                            # if avg_reward > best_reward:
                            #     print('best reward=' + str(avg_reward))
                            #     model_actor.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
                            #     model_critic.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
                            #     best_reward = avg_reward

                            current_step = 0
                            current_game_turn += 1
                            print("========================================")
                            print("Game Over!")
                            os.system('pause')

                    # ==================== Process & Update Data ====================

                    json_dict = json.loads(recv_payload)

                    # update "state"
                    observation_space = json_dict["ObservationSpace"]
                    map_flatten = observation_space["FloorMap"]["MapData"]
                    map_row = observation_space["FloorMap"]["Row"]
                    map_col = observation_space["FloorMap"]["Col"]
                    floor_map = np.array(map_flatten).reshape(map_row, map_col)
                    # map value to 0 ~ 255
                    img_map = floor_map.copy()
                    img_norm = (img_map - MAP_LABELS.min()) / (MAP_LABELS.ptp() + np.finfo(float).eps)
                    img_norm = np.around(img_norm * 255)
                    img_color = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
                    state = tf.image.resize(img_color, MOBILENET_IMG_SIZE)

                    # update "mask"
                    game_status = json_dict["GameStatus"]
                    mask = not game_status["GameFinished"]

                    # update "reward"
                    reward_struct = json_dict["Reward"]
                    reward = reward_struct["Rwrd"]
                    print(reward)

                    # ==================== Generate Game Action & Criticism ====================

                    action = model_actor.predict(
                        [np.array([state]), np.array(dummy_n), np.array(dummy_1), np.array(dummy_1), np.array(dummy_1)],
                        steps=1)
                    action = action.squeeze()
                    print(action)
                    q_value = model_critic.predict(np.array([state]), steps=1)
                    q_value = q_value.item()
                    print(q_value)

                    # ==================== Send Action to Game ====================

                    action_move = [np.around(action[0] * MAP_SHAPE[0]),
                                   np.around(action[1] * MAP_SHAPE[1])]
                    action_fire = [np.around(action[2] * MAP_SHAPE[0]),
                                   np.around(action[3] * MAP_SHAPE[1])]
                    # action_fire = [np.around(action[2] * FIRE_RANGE_MAX),
                    #     np.around(action[3] * FIRE_ANGLE_MAX)]

                    action_space = {ACTION_TYPE[0]: action_move[0], ACTION_TYPE[1]: action_move[1],
                                    ACTION_TYPE[2]: action_fire[0], ACTION_TYPE[3]: action_fire[1]}
                    send_payload = json.dumps(action_space).encode('ascii')
                    send_frame_len = len(HEADER) + FRAME_LEN_BYTES + len(send_payload) + len(EOF)
                    send_frame_len_hex = int.to_bytes(
                        send_frame_len, length=FRAME_LEN_BYTES,
                        byteorder='big', signed=True)
                    send_frame = HEADER + send_frame_len_hex + send_payload + EOF

                    conn.send(send_frame)

                    # ==================== Append Data to PPO Lists ====================

                    states.append(state)
                    actions.append(action)
                    values.append(q_value)

                    current_step += 1
