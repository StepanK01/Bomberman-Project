import os
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder as Finder
from pathfinding.core.diagonal_movement import DiagonalMovement
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
    self.input_dim = ((5, self.dimension, self.dimension),17)
    self.load_model = True
    self.path = "my-saved-model.pt"

    if (self.load_model or not self.train) and not os.path.isfile(self.path):
        raise Exception("No saved model")

    self.agent = Agent(n_actions=len(ACTIONS), input_dims=self.input_dim, lr=0.00001, log=self.logger.debug) 

    if self.load_model or not self.train:
        self.logger.info("Loading model from saved state.")
        self.agent.load_agent(self.path)

    self.crates_finder = crates_direction_finder(self.logger.debug)
    self.coins_finder = coins_direction_finder(self.logger.debug)
    self.bomb_dodge_finder = bomb_dodge_direction_finder(self.logger.debug)
    self.bomb_place_finder = bomb_place_direction_finder(self.logger.debug)
    self.others_finder = others_direction_finder(self.logger.debug)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(self, game_state)
    self.act_state = features
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
        return None

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
    crates_pos = get_pos_from_field(crates_field)

    walls_field = np.where(field == -1, 1, 0)

    danger_field = np.zeros(field.shape)
    for (xb, yb), t in bombs:
        x_p_range = 3
        x_n_range = 3
        y_p_range = 3
        y_n_range = 3
        if walls_field[xb+1,yb] == 1: x_p_range = 0
        if walls_field[xb-1,yb] == 1: x_n_range = 0
        if walls_field[xb,yb+1] == 1: y_p_range = 0
        if walls_field[xb,yb-1] == 1: y_n_range = 0
        for (i, j) in [(xb + h, yb) for h in range(-x_n_range, x_p_range+1)] + [(xb, yb + h) for h in range(-y_n_range, y_p_range+1)]:
            if (0 < i < danger_field.shape[0]-1) and (0 < j < danger_field.shape[1]-1):
                danger_field[i, j] = max(danger_field[i, j], (4-t))


    bombs_pos_field = get_field_from_pos(bombs_pos, field.shape)

    coins_field = get_field_from_pos(coins, field.shape)
    
    obstacle_field = np.where(field==-1,1,0)
    obstacle_field = np.where(explosion_map==1,1,obstacle_field)
    obstacle_field = np.where(field==1,1,obstacle_field)
    obstacle_field = np.where(danger_field==4,1,obstacle_field)
    obstacle_field = np.where(others_pos_field==1,1,obstacle_field)
    obstacle_field = np.where(bombs_pos_field==1,1,obstacle_field)

    others_pos_rfield = get_reduced_field(others_pos_field, self.dimension, agent_pos)
    coins_rfield = get_reduced_field(coins_field, self.dimension, agent_pos)
    crates_rfield = get_reduced_field(crates_field, self.dimension, agent_pos)
    danger_rfield = get_reduced_field(danger_field, self.dimension, agent_pos)
    obstacle_rfield = get_reduced_field(obstacle_field, self.dimension, agent_pos)

    path_finding_obstacle_field = np.where(obstacle_field==1,0,1)

    others_dir = self.others_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array(other_agents_pos), 0)
    bomb_place_pos, value_field = get_bomb_value_pos(obstacle_field, crates_pos, other_agents_pos, agent_pos, walls_field, num_of_targets=5)
    bomb_place_dir = self.bomb_place_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array(bomb_place_pos), 0, value_field)
    coins_dir = self.coins_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array(coins), 5)

    bomb_dodge_dir = get_bomb_dodge_dir(path_finding_obstacle_field, danger_field, agent_pos, self, preprocess=5)
    

    features = np.concatenate((others_dir,coins_dir,bomb_dodge_dir,bomb_place_dir))
    vision_field = np.stack((others_pos_rfield,coins_rfield,crates_rfield,danger_rfield,obstacle_rfield))

    return vision_field, features

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


def get_pos_from_field(field, value=1):
    indicies = np.where(field==value)
    if len(indicies[0])!=0:
        pos = [None]*len(indicies[0])
        for i in range(len(indicies[0])):
            pos[i] = (indicies[0][i],indicies[1][i])
    else:
        pos = []
    return pos

def get_bomb_dodge_dir(obstacle_field, danger_field, pos, self, preprocess):
    actions = np.zeros(4)
    if danger_field[pos[0],pos[1]] != 0:
        no_save_square = (1 - obstacle_field) + danger_field
        save_field = np.where(no_save_square==0,1,0) 
        save_pos = get_pos_from_field(save_field)
        actions = self.bomb_dodge_finder.get_direction_of_shortest_path(obstacle_field, pos, np.array(save_pos),preprocess,danger_field) 

    return actions

def get_bomb_value_pos(obstacle_field, crates_pos, others_pos, pos, walls_field, num_of_targets):
    value_field = np.zeros(obstacle_field.shape)
    for (xc, yc) in crates_pos:
        x_p_range = 3
        x_n_range = 3
        y_p_range = 3
        y_n_range = 3
        if walls_field[xc+1,yc] == 1: x_p_range = 0
        if walls_field[xc-1,yc] == 1: x_n_range = 0
        if walls_field[xc,yc+1] == 1: y_p_range = 0
        if walls_field[xc,yc-1] == 1: y_n_range = 0
        for (i, j) in [(xc + h, yc) for h in range(-x_n_range, x_p_range+1)] + [(xc, yc + h) for h in range(-y_n_range, y_p_range+1)]:
            if (0 < i < value_field.shape[0]-1) and (0 < j < value_field.shape[1]-1):
                value_field[i, j] += 1

    for (xo, yo) in others_pos:
        x_p_range = 3
        x_n_range = 3
        y_p_range = 3
        y_n_range = 3
        if walls_field[xo+1,yo] == 1: x_p_range = 0
        if walls_field[xo-1,yo] == 1: x_n_range = 0
        if walls_field[xo,yo+1] == 1: y_p_range = 0
        if walls_field[xo,yo-1] == 1: y_n_range = 0
        for (i, j) in [(xo + h, yo) for h in range(-x_n_range, x_p_range+1)] + [(xo, yo + h) for h in range(-y_n_range, y_p_range+1)]:
            if (0 < i < value_field.shape[0]-1) and (0 < j < value_field.shape[1]-1):
                value_field[i, j] += 2

    pre_dist_value_field = value_field.copy()

    for x in range(value_field.shape[0]):
        for y in range(value_field.shape[1]):
            dist = (np.abs((x-pos[0])/2)+np.abs((y-pos[1])/2))**2 
            value_field[x,y] -= dist


    value_field = np.where(obstacle_field!=0,0,value_field) 

    num_good_places = (value_field > 0).sum()
    if num_good_places == 0:
        value_field = pre_dist_value_field.copy()
        for x in range(value_field.shape[0]):
            for y in range(value_field.shape[1]):
                dist = (np.abs((x-pos[0])/12)+np.abs((y-pos[1])/12))**2 
                value_field[x,y] -= dist
        num_good_places = (value_field > 0).sum()

    if num_good_places < num_of_targets:
        num_of_targets = num_good_places

    value_pos_index = np.unravel_index(np.argpartition(value_field,-num_of_targets,axis=None), value_field.shape)
    value_pos = [None]*num_of_targets
    for i in range(num_of_targets):
        value_pos[i] = (value_pos_index[0][-i-1],value_pos_index[1][-i-1])

    return value_pos, pre_dist_value_field

class direction_finder:
    def __init__(self, logger):
        self.default_action = np.zeros(4)
        self.log = logger

    def preprocessing(self, pos, targets_pos, num):
        if num >= len(targets_pos):
            num = 0
        if num != 0:
            rel_pos = np.abs(targets_pos - pos)
            dists = rel_pos.sum(axis=1)
            indicies = np.argpartition(dists, num)[:num]
            targets_pos = targets_pos[indicies]
        return targets_pos

    def get_path_to_target(self, field, pos, target_pos):
        field[pos[0],pos[1]] = 1
        grid = Grid(matrix=field.T) 
        start = grid.node(pos[0],pos[1])
        end = grid.node(target_pos[0],target_pos[1])
        finder = Finder(diagonal_movement=DiagonalMovement.never)
        path, _ = finder.find_path(start, end, grid)
        path_len = len(path)
        return path, path_len

    def get_direction_from_path(self, path):
        action = self.default_action.copy()
        dif = np.array(path[1])-np.array(path[0])
        if dif[1] < 0:
            action[0] = 1 #UP
        elif dif[1] > 0:
            action[2] = 1 #DOWN
        elif dif[0] > 0:
            action[1] = 1 #RIGHT
        elif dif[0] < 0:
            action[3] = 1 #LEFT
        return action

    def get_direction_of_shortest_path(self, field:np.array, pos:tuple, targets_pos:np.array, num:int, *args):
        if len(targets_pos)==0:
            return self.default_action

        targets_pos = self.preprocessing(pos, targets_pos, num)

        paths = [None]*len(targets_pos)
        path_lens = np.empty(len(targets_pos))
        for i, target_pos in enumerate(targets_pos):
            path, path_len = self.get_path_to_target(field, pos, target_pos)
            paths[i] = path
            path_lens[i] = self.adjust_path_len(path, path_len, *args)

            return_action, action = self.check_single_path_for_action(path, path_len, *args) 
            if return_action:
                return action
        return_action, action = self.check_paths_for_action(field, pos, num, paths, path_lens, *args)
        if return_action:
            return action

        index = np.argmin(path_lens)
        path = paths[index]

        return self.get_direction_from_path(path)

    def adjust_path_len(self, path, path_len, *args):
        if path == []:
            path_len = 1000
        return path_len

    def check_single_path_for_action(self, path, path_len, *args):
        return False, None

    def check_paths_for_action(self, field, pos, num, paths, path_lens, *args):
        if (path_lens != 1000).sum() == 0:
            return True, self.default_action
        else:
            return False, None


class crates_direction_finder(direction_finder):
    def get_path_to_target(self, field, pos, target_pos):
        field[target_pos[0],target_pos[1]] = 1
        return super().get_path_to_target(field, pos, target_pos)

    def check_single_path_for_action(self, path, path_len):
        if path_len == 2:
            action = self.default_action
            return True, action
        else:
            return super().check_single_path_for_action(path, path_len)

class others_direction_finder(direction_finder):
    def get_path_to_target(self, field, pos, target_pos):
        field[target_pos[0],target_pos[1]] = 1
        return super().get_path_to_target(field, pos, target_pos)

    def check_single_path_for_action(self, path, path_len, *args):
        if path_len == 2:
            return True, self.default_action
        else:
            return super().check_single_path_for_action(path, path_len, *args)


class coins_direction_finder(direction_finder):
    def adjust_path_len(self, path, path_len):
        path_len = super().adjust_path_len(path, path_len)
        if path_len == 1:
            path_len = 1000
        return path_len


class bomb_dodge_direction_finder(direction_finder):
    def __init__(self, logger):
        super().__init__(logger)
        self.check_dangerlevel = 1

    def adjust_path_len(self, path, path_len, dangerfield):
        path_len = super().adjust_path_len(path, path_len)
        for j, tile in enumerate(path):
            if j > 4 - dangerfield[tile[0],tile[1]] and dangerfield[tile[0],tile[1]] != 0 and j - (4 - dangerfield[tile[0],tile[1]]) < 3: #TODO Warten kann auch eine gute Idee sein.
                path_len = 1000
        return path_len

    def check_paths_for_action(self, field, pos, num, paths, path_lens, dangerfield):
        no_path_available, action = super().check_paths_for_action(field, pos, num, paths, path_lens, dangerfield)
        if no_path_available and not self.check_dangerlevel >= dangerfield[pos]:
            targets_pos = get_pos_from_field(dangerfield, value=self.check_dangerlevel)
            self.check_dangerlevel += 1
            action = self.get_direction_of_shortest_path(field, pos, np.array(targets_pos), num, dangerfield)
            self.check_dangerlevel = 1
            return True, action
        else:
            self.check_dangerlevel = 1
            return no_path_available, action 

class bomb_place_direction_finder(direction_finder):
    def __init__(self, logger):
        super().__init__(logger)
        self.default_action = np.zeros(5)

    def adjust_path_len(self, path, path_len, value_field):
        path_len = super().adjust_path_len(path, path_len)
        if path_len != 1000:
            path_len = - value_field[path[-1]] + (path_len/4)**2
        return path_len

    def get_direction_from_path(self, path):
        if len(path) == 1:
            action = self.default_action.copy()
            action[4] = 1
            return action
        else:
            return super().get_direction_from_path(path)

    def check_paths_for_action(self, field, pos, num, paths, path_lens, *args):
        no_path_available, action = super().check_paths_for_action(field, pos, num, paths, path_lens, *args)
        if no_path_available:
            return no_path_available, action
        else:
            return no_path_available, action