import os

import numpy as np

from .model import Agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up")

    self.dimension = 9
    self.input_dim = 405
    self.load_model = False
    self.path = "my-saved-model.pt"

    if (self.load_model or not self.train) and not os.path.isfile(self.path):
        raise Exception("No saved model")

    self.agent = Agent(n_actions=len(ACTIONS), input_dims=[self.input_dim], lr=0.001, log=self.logger.debug)

    if self.load_model or not self.train:
        self.logger.info("Loading model from saved state.")
        self.agent.load_agent(self.path)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(self, game_state)
    self.logger.debug("Querying model for action.")
    if self.train:
        action_ = self.agent.choose_train_action(features)
        action = ACTIONS[action_]
        return action
    else:
        action_ = self.agent.choose_action(features)
        action = ACTIONS[action_]
        return action


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return np.zeros(self.input_dim)

    _, _, field, agent_input, others, bombs, coins, _, explosion_map = game_state.values()
    _, agent_score, agent_has_bomb, agent_pos = agent_input


    num_of_others = len(others)
    other_agents_score = np.empty(num_of_others)
    other_agents_has_bomb = np.empty(num_of_others)
    other_agents_pos = list(np.empty(num_of_others))
    for i, other in enumerate(others):
        _, other_agents_score[i], other_agents_has_bomb[i], other_agents_pos[i] = other

    bombs_pos = list(np.empty(len(bombs)))
    for i, bomb in enumerate(bombs):
        bombs_pos[i], _ = bomb



    others_pos_field = get_field_from_pos(other_agents_pos, field.shape)
    crates_field = np.where(field==1,1,0)

    danger_field = np.ones(field.shape) * 0
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 3+1)] + [(xb, yb + h) for h in range(-3, 3+1)]:
            if (0 < i < danger_field.shape[0]) and (0 < j < danger_field.shape[1]):
                danger_field[i, j] = max(danger_field[i, j], (4-t))

    bombs_pos_field = get_field_from_pos(bombs_pos, field.shape)

    coins_field = get_field_from_pos(coins, field.shape)
    
    obstacle_field = np.where(field==-1,1,0)
    obstacle_field = np.where(explosion_map==1,1,obstacle_field)
    obstacle_field = np.where(field==1,1,obstacle_field)
    obstacle_field = np.where(danger_field==3,1,obstacle_field)
    obstacle_field = np.where(danger_field==4,1,obstacle_field)
    obstacle_field = np.where(others_pos_field==1,1,obstacle_field)
    obstacle_field = np.where(bombs_pos_field==1,1,obstacle_field)

    others_pos_rfield = get_reduced_field(others_pos_field, self.dimension, agent_pos)
    coins_rfield = get_reduced_field(coins_field, self.dimension, agent_pos)
    crates_rfield = get_reduced_field(crates_field, self.dimension, agent_pos)
    danger_rfield = get_reduced_field(danger_field, self.dimension, agent_pos)
    obstacle_rfield = get_reduced_field(obstacle_field, self.dimension, agent_pos)

    features_rfield = np.stack((others_pos_rfield, coins_rfield, crates_rfield, danger_rfield, obstacle_rfield))
    features = features_rfield.flatten()
    return features

def get_field_from_pos(positions: np.array, field_dimension: tuple, new_value=1, standard_value=0):
    field = np.ones(field_dimension)*standard_value
    for pos in positions:
        field[pos] = new_value
    return field


def get_reduced_field(field: np.array, dimension: int, position: tuple, standard_value=0) -> np.array:
    extra_space = (np.floor(dimension/2)-1).astype(int)
    bigger_dimension = extra_space*2 + field.shape[0]
    bigger_field = np.ones((bigger_dimension,bigger_dimension))*standard_value
    bigger_field[extra_space:extra_space+field.shape[0],extra_space:extra_space+field.shape[1]] = field
    reduced_field = bigger_field[position[0]-1:position[0]-1+dimension,position[1]-1:position[1]-1+dimension]
    return reduced_field
