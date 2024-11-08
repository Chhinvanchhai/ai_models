# Training Loop Breakdown

### Training Loop Breakdown

1.  **Initialize Parameters**
    - `episodes`: Number of games (episodes) the agent will play to learn.
    - `gamma` **(discount factor)**: Controls the importance of future rewards. A value close to 1.0 makes the agent consider future rewards more heavily.
    - `epsilon` **(exploration rate)**: Controls the balance between exploration (random moves) and exploitation (choosing the best-known move). We start with a high epsilon (e.g., 1.0) to encourage exploration and gradually decrease it to favor exploitation as the agent learns.
    - `epsilon_decay`: Rate at which `epsilon` decreases after each episode, encouraging the model to exploit learned strategies over time.
    - `epsilon_min`: Minimum value of epsilon to prevent it from becoming zero.
    - `batch_size`: Size of the random sample batch used for training at each step.
2.  **Replay Memory**
    - **Experience replay** is a key component of DQN, allowing the model to learn from past experiences. A **deque** (double-ended queue) is used to store experiences (state, action, reward, next state, and done status) for a large number of moves.
    - Each episode (or game) generates experiences, and the deque helps retain the most recent experiences. When enough experiences are stored, we sample them randomly to train the model.
3.  **Start the Game (Episode)**
    - For each episode, we reset the Tic-Tac-Toe environment (the board) to a starting state. The agent’s goal is to play a full game from this initial state.
4.  **Action Selection with Epsilon-Greedy Policy**
    - The agent chooses an action based on the **epsilon-greedy policy**:
        - **Exploration**: With probability `epsilon`, the agent makes a random move (explores the environment).
        - **Exploitation**: With probability `1 - epsilon`, the agent chooses the move with the highest Q-value (the best-known move so far).
    - Over time, as `epsilon` decreases, the agent shifts from exploration to exploitation.
5.  **Take Action and Observe Outcome**
    - The selected action is applied to the environment with `env.step(action, player)`.
    - This step returns:
        - `next_state`: The new board state after the action.
        - `reward`: Positive for a win, negative for a loss, and zero otherwise.
        - `done`: Indicates if the game is over (either due to a win/loss or a full board).
6.  **Store Experience in Replay Memory**
    - Each (state, action, reward, next state, done) experience is added to the replay memory. This allows the model to revisit and learn from these experiences in a random order.
7.  **Sample a Batch and Train the Model**
    - If the replay memory has enough experiences (more than `batch_size`), we sample a batch and train the model on it:
        - For each experience in the batch:
            - If the game is done (`done=True`), the target Q-value is simply the reward.
            - If the game isn’t over, we estimate the **future Q-value** by adding the reward to the discounted maximum Q-value of the next state (i.e., `reward + gamma * max(Q(next_state))`).
        - The model is trained to update the Q-values for the selected action, bringing it closer to the target value calculated above.
    - This training step is performed in batches, so the model updates based on multiple experiences simultaneously.
8.  **Decrease Epsilon**
    - After each episode, decrease `epsilon` by multiplying it with `epsilon_decay` until it reaches `epsilon_min`. This gradual decrease in `epsilon` helps the agent shift from exploring to exploiting its learned strategies.
9.  **Print Progress**
    - Print progress messages, such as the current episode and epsilon, to monitor training.