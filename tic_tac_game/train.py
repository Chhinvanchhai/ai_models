from collections import deque
import tic_tac_toe_env 
import tensorflow as tf
import numpy as np
import random

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(9,), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9)  # Outputs a Q-value for each of the 9 board positions
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = create_model()



def train_dqn(episodes=1000, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=32):
    memory = deque(maxlen=2000)
    env = tic_tac_toe_env.TicTacToeEnv()
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 9])
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = random.choice(env.available_actions())
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0][env.available_actions()])

            # Take the action
            next_state, reward, done = env.step(action, 1)
            next_state = np.reshape(next_state, [1, 9])
            # Store experience in replay memory
            memory.append((state, action, reward, next_state, done))
            state = next_state
            print(f"------start:{state}")

            # Sample a batch from memory and train
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target += gamma * np.amax(model.predict(ns, verbose=0)[0])
                    target_f = model.predict(s, verbose=0)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {episode + 1}/{episodes} - Epsilon: {epsilon:.2f}")
        # Save the trained model
        model.save("tic_tac_toe_dqn_model.keras")


train_dqn()
