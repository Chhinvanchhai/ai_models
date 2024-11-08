import tensorflow as tf
import numpy as np
import tic_tac_toe_env 

# Load the saved model
model = tf.keras.models.load_model("tic_tac_toe_dqn_model.keras", 
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})


# Initialize the game environment
env = tic_tac_toe_env.TicTacToeEnv()


def get_model_action(state):
    """Get the model's best action for the current state."""
    q_values = model.predict(state, verbose=0)
    valid_actions = env.available_actions()
    # Choose the action with the highest Q-value among available actions
    best_action = max(valid_actions, key=lambda x: q_values[0][x])
    return best_action

def play_against_model():
    """Function to play a game between a human and the model."""
    state = env.reset()
    state = np.reshape(state, [1, 9])
    done = False

    print("Starting Tic-Tac-Toe game! You are Player -1.")

    while not done:
        # Display the board
        print("Current board:")
        print(env.board)

        # Player's turn
        player_move = int(input("Enter your move (0-8): "))
        if player_move not in env.available_actions():
            print("Invalid move. Try again.")
            continue

        # Apply player move
        state, reward, done = env.step(player_move, -1)  # Player is -1
        state = np.reshape(state, [1, 9])

        # Check if the player has won or if the game is a draw
        if done:
            if reward == -1:
                print("You win!")
                print(env.board)
            else:
                print("It's a draw!")
            break

        # Model's turn
        model_move = get_model_action(state)
        state, reward, done = env.step(model_move, 1)  # Model is 1
        print(f"Model chose move: {model_move}")

        # Check if the model has won
        if done:
            if reward == 1:
                print(env.board)
                print("Model wins!")
            else:
                print("It's a draw!")
            break

# Run the game
play_against_model()
