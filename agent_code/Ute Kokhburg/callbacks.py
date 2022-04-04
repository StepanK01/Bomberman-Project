import os
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder as Finder
from pathfinding.core.diagonal_movement import DiagonalMovement
import numpy as np

from .model import Agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MAX_COINS = 9


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.


    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up")

    self.dimension = 9
    self.input_dim = [8]
    self.load_model = False
    self.path = "my-saved-model.pt"

    if (self.load_model or not self.train) and not os.path.isfile(self.path):
        raise Exception("No saved model")

    self.agent = Agent(n_actions=3, input_dims=self.input_dim, lr=0.0001, log=self.logger.debug) # lr=0.0001

    if self.load_model or not self.train:
        self.logger.info("Loading model from saved state.")
        self.agent.load_agent(self.path)

    self.coins_finder = coins_direction_finder(self.logger.debug)
    self.others_finder = others_direction_finder(self.logger.debug)
    self.bomb_place_finder = bomb_place_direction_finder(self.logger.debug)
    self.bomb_dodge_finder = bomb_dodge_direction_finder(self.logger.debug)

    self.finders = [self.coins_finder, self.others_finder, self.bomb_place_finder, self.bomb_dodge_finder]
    
    self.bomb_values = bomb_place_values(self.logger.debug)

    self.coins = []
    self.num_coins = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Reset the coin count
    if game_state["step"] == 1:
        self.num_coins = 0

    #To avoid throwing any error in the tournement (we did not discovere any ones in our extensive testing, but we just wanted to make sure it would not gone happen)
    try:
        features = state_to_features(self, game_state)
        self.act_state = features
        weigths, actions = spilt_into_weights_and_actions(self, features)
        self.logger.debug("Querying model for action.")

        if self.train:
            index = self.agent.choose_train_action(weigths)
            self.action_index = index
        else:
            index = self.agent.choose_action(weigths)

        action = actions[index]
        self.logger.info("choose")
        if action == None:
            action = "WAIT"
        self.logger.info(action)
    except:
        action = "WAIT"
    return action

# This function serves as a communicator between the agents model and the game and state_to_features funciton
def spilt_into_weights_and_actions(self, features):
    num = len(self.finders)
    actions = [None] * num 
    weights = np.empty(num*2)
    feature_index = 0
    weigths_index = 0
    for i, finder in enumerate(self.finders):
        actions[i] = finder.last_recomendation
        feature_length = len(finder.default_action)
        feature_index += feature_length
        weights[weigths_index] = features[feature_index-1]
        weigths_index += 1
        weights[weigths_index] = features[feature_index-2]
        weigths_index += 1
    actions = np.array(actions)
    actions = np.delete(actions, 1)
    return weights, actions


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    # Extract relevant information from the gamestate and reformat it
    _, _, field, agent_input, others, bombs, coins, _, explosion_map = game_state.values()
    _, agent_score, agent_has_bomb, agent_pos = agent_input

    for coin in coins:
        if coin not in self.coins:
            self.coins.append(coin)
            self.num_coins += 1

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
    self.num_crates = len(crates_pos)

    walls_field = np.where(field == -1, 1, 0)

    danger_field = get_dangerfield(bombs, walls_field)

    bombs_pos_field = get_field_from_pos(bombs_pos, field.shape)
    
    # Creating the obstacle_field
    # Not passsable positions (obstacles, explosions, in the next step occouring explosions, crates, placed bombs, opponents)

    radius = 6

    obstacle_field_reduced = np.where(field==-1,1,0)
    obstacle_field_reduced = np.where(get_reduce_field_to_neighborhood(explosion_map, agent_pos, radius)==1,1,obstacle_field_reduced)
    obstacle_field_reduced = np.where(field==1,1,obstacle_field_reduced)
    obstacle_field_reduced = np.where(get_reduce_field_to_neighborhood(danger_field, agent_pos, radius)==4,1,obstacle_field_reduced)
    obstacle_field_reduced = np.where(get_reduce_field_to_neighborhood(others_pos_field, agent_pos, radius)==1,1,obstacle_field_reduced)
    obstacle_field_reduced = np.where(get_reduce_field_to_neighborhood(bombs_pos_field, agent_pos, radius)==1,1,obstacle_field_reduced)

    path_finding_obstacle_field = np.where(obstacle_field_reduced==1,0,1)

    # Calculaing the features
    coins_dir = self.coins_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array(coins), 5, self.others_finder, np.array(other_agents_pos))
    others_dir = self.others_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array(other_agents_pos), 0)
    self.bomb_values.calc_value_field(agent_pos, crates_pos, other_agents_pos, walls_field, obstacle_field_reduced, danger_field, self.num_coins, self.num_crates) #TODO eventuell nicht reduced
    bomb_place_dir = self.bomb_place_finder.get_direction_of_shortest_path(path_finding_obstacle_field, agent_pos, np.array([None]), 5, self.bomb_values)
    bomb_dodge_dir = get_bomb_dodge_dir(path_finding_obstacle_field, danger_field, agent_pos, self, 5, bomb_place_dir, others_dir, coins_dir)
    
    features = np.concatenate((coins_dir,others_dir,bomb_place_dir,bomb_dodge_dir))
    return features

def get_dangerfield(bombs, walls_field):
    danger_field = np.zeros(walls_field.shape)
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
    return danger_field

def get_reduce_field_to_neighborhood(field, pos, radius):
    """
    Returns a field with the values of the given field
    for every square which is a Manhattan-distance radius away from pos
    and zero for every other square
    """
    x, y = pos
    red_field = np.zeros(field.shape)
    for i in range(-radius,radius+1):
        for j in range(-radius+np.abs(i),radius-np.abs(i)+1):
            l = i + x
            k = j + y
            if (0 <= l < red_field.shape[0]) and (0 <= k < red_field.shape[1]):
                red_field[l,k] = field[l,k]
    return red_field


def get_field_from_pos(positions: np.array, field_dimension: tuple, new_value=1, standard_value=0):
    field = np.ones(field_dimension)*standard_value
    for pos in positions:
        field[pos] = new_value
    return field

def get_pos_from_field(field, value=1):
    indicies = np.where(field==value)
    if len(indicies[0])!=0:
        pos = [None]*len(indicies[0])
        for i in range(len(indicies[0])):
            pos[i] = (indicies[0][i],indicies[1][i])
    else:
        pos = []
    return pos

def get_bomb_dodge_dir(obstacle_field, danger_field, pos, self, preprocess, bomb_place_dir, others_dir, coins_dir):
    actions = self.bomb_dodge_finder.default_action
    #Only ask the feature if in danger
    if danger_field[pos[0],pos[1]] != 0:
        no_save_square = (1 - obstacle_field) + danger_field
        save_field = np.where(no_save_square==0,1,0)
        save_pos = get_pos_from_field(save_field)
        actions = self.bomb_dodge_finder.get_direction_of_shortest_path(obstacle_field, pos, np.array(save_pos),preprocess,danger_field, bomb_place_dir, others_dir, coins_dir) 
    else:
        self.bomb_dodge_finder.last_recomendation = None

    return actions

def distance_from_pos(pos, target_pos):
    return np.abs(pos[0]-target_pos[0]) + np.abs(pos[1]-target_pos[1]) # Manhattan distance

def distance_value_adjustment(value, pos_dif, dif = 4, pot = 2):
    return value - (pos_dif/dif)**pot

class bomb_place_values:
    def __init__(self, logger):
        self.log = logger
        self.dims = None
        self.value_field = None
        self.temp_value_field = None
        self.pre_dist_value_field = None
        self.temp_pre_dist_value_field = None
        self.requested_num = 0
        self.num_good_places = 0
        self.already_adjusted = False
        self.pos = None

    def calc_value_field(self, pos, crates_pos, others_pos, walls_field, obstacle_field, dangerfield, num_coins, num_crates):
        self.requested_num = 0
        self.dims = obstacle_field.shape
        self.pre_dist_value_field = self.get_crates_value_field(crates_pos, walls_field, num_coins, num_crates) + self.get_others_value_field(pos, others_pos, walls_field, obstacle_field, dangerfield)
        self.pre_dist_value_field = np.where(obstacle_field == 1,-1000,self.pre_dist_value_field)
        self.temp_pre_dist_value_field = self.pre_dist_value_field.copy()
        self.value_field = self.pre_dist_value_field.copy()
        self.value_field = self.adjust_for_distance(self.value_field, pos)
        self.num_good_places = (self.value_field > 0).sum()
        self.temp_value_field = self.value_field.copy()
        self.pos = pos

    def adjust_for_distance(self, field, pos, dif = 4, pot = 2):
        field = field.copy()
        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                field[x,y] = distance_value_adjustment(field[x,y], distance_from_pos(pos,(x,y)), dif = dif, pot = pot)
        return field

    def get_crates_value_field(self, crates_pos, walls_field, num_coins, num_crates):
        self.value_field = np.zeros(self.dims)
        if num_coins >= MAX_COINS or num_crates == 0:
            return self.value_field
        factor = (118/9)*(MAX_COINS - num_coins)/num_crates # Approximatly 1 in the beginning
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
                if (0 < i < self.value_field.shape[0]-1) and (0 < j < self.value_field.shape[1]-1):
                    self.value_field[i, j] += factor
        return self.value_field

    def get_others_value_field(self, pos, others_pos, walls_field, obstacle_field, dangerfield):
        self.value_field = np.zeros(self.dims)
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
                if (0 < i < self.value_field.shape[0]-1) and (0 < j < self.value_field.shape[1]-1):
                    if (np.array(pos)-np.array([xo,yo])).sum() <= 6:
                        self.value_field[i, j] += self.get_danger_of_other(pos, (xo,yo), (i,j), walls_field, obstacle_field, dangerfield)
                    else:
                        self.value_field[i, j] = 0.5
        return self.value_field

    def get_danger_of_other(self, pos, other_pos, bomb_pos, walls_field, obstacle_field, dangerfield):
        obstacle_field = obstacle_field.copy() 
        obstacle_field[bomb_pos] = 1
        reachable_pos = self.get_reachable_pos(other_pos, obstacle_field)
        own_bomb_dangerfield = get_dangerfield([(bomb_pos,3)], walls_field)
        own_danger_pos = get_pos_from_field(own_bomb_dangerfield, value = 1)
        if pos == bomb_pos:
            other_danger_pos = get_pos_from_field(dangerfield, value=1)
            danger_pos = own_danger_pos + other_danger_pos
        else:
            danger_pos = own_danger_pos
        dangerless_pos = []
        for (r_pos, step) in reachable_pos:
            if r_pos not in danger_pos:
                dangerless_pos.append(r_pos) 

        score = 10*np.exp(-len(dangerless_pos)/4)
        score = max(score, 0.5)

        return score

    def get_reachable_pos(self, other_pos, obstacle_field):
        reachable_pos = [(other_pos,0)]
        already_checked_pos = []
        for (r_pos,step) in reachable_pos:
            dirs = np.array([(1,0),(-1,0),(0,1),(0,-1)])
            if step <= 3:
                if r_pos not in already_checked_pos:
                    already_checked_pos.append(r_pos)
                    for dir in dirs:
                        neighbor_pos = tuple(r_pos+dir)
                        if neighbor_pos not in already_checked_pos:
                            if obstacle_field[neighbor_pos] == 0:
                                reachable_pos.append((neighbor_pos,step+1))
        return reachable_pos

    def get_next_value_pos(self, num):
        if self.num_good_places > self.requested_num:
            if self.num_good_places - self.requested_num < num:
                num = self.num_good_places - self.requested_num
            self.requested_num += num
            value_pos_index = np.unravel_index(np.argpartition(self.temp_value_field,-num,axis=None), self.temp_value_field.shape)
            value_pos = [None]*num
            for i in range(num):
                value_pos[i] = (value_pos_index[0][-i-1],value_pos_index[1][-i-1])
                self.temp_value_field[value_pos[i]] = -1000
                self.temp_pre_dist_value_field[value_pos[i]] = -1000
            return value_pos
        else:
            # Adjust the distance penalty if no good squares are left
            if not self.already_adjusted:
                self.temp_value_field = self.adjust_for_distance(self.temp_pre_dist_value_field, self.pos, dif = 12, pot = 2)
                self.num_good_places += (self.temp_value_field > 0).sum()
                self.already_adjusted = True
                return self.get_next_value_pos(num)
            else:
                self.already_adjusted = False
                return None


class direction_finder:
    def __init__(self, logger):
        self.default_action = np.zeros(6)
        self.log = logger
        self.last_recomendation = None

    def preprocessing(self, field, pos, targets_pos, num):
        if num >= len(targets_pos):
            num = 0
        if num != 0:
            rel_pos = np.abs(targets_pos - pos)
            dists = rel_pos.sum(axis=1)
            indicies = np.argpartition(dists, num)[:num]
            targets_pos = targets_pos[indicies]
        return targets_pos

    def get_path_to_target(self, field, pos, target_pos):
        field = field.copy()
        field[pos[0],pos[1]] = 1
        grid = Grid(matrix=field.T) # The transposing must be done since the library works with a different coordinate system
        start = grid.node(pos[0],pos[1])
        end = grid.node(target_pos[0],target_pos[1])
        finder = Finder(diagonal_movement=DiagonalMovement.never)
        path, _ = finder.find_path(start, end, grid)
        path_len = len(path)
        return path, path_len

    def get_direction_from_path(self, path, path_len, field, *args):
        action = self.default_action.copy()
        dif = np.array(path[1])-np.array(path[0])
        if dif[1] < 0:
            action[0] = 1 #UP
            self.last_recomendation = "UP"
        elif dif[1] > 0:
            action[2] = 1 #DOWN
            self.last_recomendation = "DOWN"
        elif dif[0] > 0:
            action[1] = 1 #RIGHT
            self.last_recomendation = "RIGHT"
        elif dif[0] < 0:
            action[3] = 1 #LEFT
            self.last_recomendation = "LEFT"
        action[-1] = path_len
        action[-2] = 1
        return action

    def get_direction_of_shortest_path(self, field:np.array, pos:tuple, targets_pos:np.array, num:int, *args):
        if len(targets_pos)==0:
            self.last_recomendation = None
            return self.default_action

        targets_pos = self.preprocessing(field, pos, targets_pos, num)

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
        path_len = path_lens[index]

        return self.get_direction_from_path(path, path_len, field, *args)

    def adjust_path_len(self, path, path_len, *args):
        if path == []:
            path_len = 1000
        return path_len

    def check_single_path_for_action(self, path, path_len, *args):
        return False, None

    def check_paths_for_action(self, field, pos, num, paths, path_lens, *args):
        if (path_lens != 1000).sum() == 0:
            self.last_recomendation = None
            return True, self.default_action
        else:
            return False, None


class others_direction_finder(direction_finder):
    def get_path_to_target(self, field, pos, target_pos):
        # Make target walkable
        field = field.copy()
        field[target_pos[0],target_pos[1]] = 1
        return super().get_path_to_target(field, pos, target_pos)

    def check_single_path_for_action(self, path, path_len, *args):
        if path_len == 2:
            self.last_recomendation = None
            action = self.default_action.copy()
            action[-1] = path_len
            action[-2] = 1
            return True, self.default_action
        else:
            return super().check_single_path_for_action(path, path_len, *args)


class coins_direction_finder(direction_finder): 
    def adjust_path_len(self, path, path_len, *args):
        path_len = super().adjust_path_len(path, path_len, *args)
        # Cause the gamstate sometimes still shows a coin even tough the agent is on top of it
        if path_len == 1:
            path_len = 1000
        return path_len

    def get_direction_from_path(self, path, path_len, field, others_finder:others_direction_finder, others_pos):
        coin_to_others_action = others_finder.get_direction_of_shortest_path(field, path[-1], np.array(others_pos), 0)
        path_len = path_len - coin_to_others_action[-1] # How much closer to oneself, compared to the rest
        return super().get_direction_from_path(path, path_len, field, others_finder, others_pos)


class bomb_dodge_direction_finder(direction_finder):
    def __init__(self, logger):
        super().__init__(logger)
        self.checked_transitions = False
        self.default_action = np.zeros(7)

    def get_direction_of_shortest_path(self, field: np.array, pos: tuple, targets_pos: np.array, num: int, dangerfield, bomb_place_dir, others_dir, coins_dir):
        if len(targets_pos)==0:
            self.last_recomendation = "WAIT"
            action = self.default_action.copy()
            action[4] = 1
            action[-1] = 0
            action[-2] = 1
            return action
        return super().get_direction_of_shortest_path(field, pos, targets_pos, num, dangerfield, bomb_place_dir, others_dir, coins_dir)

    def adjust_path_len(self, path, path_len, dangerfield, bomb_place_dir, others_dir, coins_dir):
        path_len = super().adjust_path_len(path, path_len, dangerfield, bomb_place_dir, others_dir, coins_dir)
        # Check if the path is walkable without exploding
        for j, tile in enumerate(path):
            if j > 4 - dangerfield[tile[0],tile[1]] and dangerfield[tile[0],tile[1]] != 0 and j - (4 - dangerfield[tile[0],tile[1]]) < 3:
                path_len = 1000
        if path_len != 1000:
            action = self.get_direction_from_path(path, path_len, None, dangerfield)
            if others_dir[-1] <= 5 and (others_dir[:4] == action[:4]).all():
                path_len += 0.4
            if coins_dir[-1] <= 6 and (coins_dir[:4] == action[:4]).all():
                path_len += -0.2
            if (bomb_place_dir[:4] == action[:4]).all():
                path_len += -0.1
        return path_len

    def check_paths_for_action(self, field, pos, num, paths, path_lens, dangerfield, bomb_place_dir, others_dir, coins_dir):
        # Try to walk to border in the dangerlevelfield
        no_path_available, action = super().check_paths_for_action(field, pos, num, paths, path_lens, dangerfield, bomb_place_dir, others_dir, coins_dir)
        if no_path_available and not self.checked_transitions:
            self.checked_transitions = True
            transitions_pos = self.get_dangerfield_transitions(field, pos, dangerfield)
            action = self.get_direction_of_shortest_path(field, pos, np.array(transitions_pos), 0, dangerfield, bomb_place_dir, others_dir, coins_dir)
            self.checked_transitions = False
            return True, action
        else:
            self.check_dangerlevel = False
            action = self.default_action.copy()
            action[4] = 1
            action[-1] = 0
            action[-2] = 1
            self.last_recomendation = "WAIT"
            return no_path_available, action

    def get_direction_from_path(self, path, path_len, field, dangerfield, *args):
        if len(path) == 1:
            action = self.default_action.copy()
            action[4] = 1
            action[-1] = 0
            action[-2] = 1
            self.last_recomendation = "WAIT"
            return action
        else:
            index = np.array(path)
            explosion_time = 4 - dangerfield[index[:,0],index[:,1]]
            time_dif = explosion_time - np.arange(len(path))
            path_len = np.min(time_dif)
            return super().get_direction_from_path(path, path_len, field, dangerfield, *args)

    def get_dangerfield_transitions(self, field, pos, dangerfield):
        own_danger = dangerfield[pos]
        # Transitions are sorted by dangerleveldifference
        transitions_pos = ([],[],[],[])
        check_positions = get_pos_from_field(dangerfield, value=own_danger)
        # Obstacles are Zeros
        check_field = np.where(field == 0, own_danger, dangerfield)
        for check_pos in check_positions:
            dirs = np.array([(1,0),(-1,0),(0,1),(0,-1)])
            for dir in dirs:
                neighbor_pos = tuple(check_pos+dir)
                if check_field[neighbor_pos] != own_danger:
                    if dangerfield[check_pos] < dangerfield[neighbor_pos]:
                        danger_level_difference = int(dangerfield[neighbor_pos]-dangerfield[check_pos])
                        transitions_pos[danger_level_difference-1].append(check_pos)
                    else:
                        danger_level_difference = int(dangerfield[check_pos]-dangerfield[neighbor_pos])
                        transitions_pos[danger_level_difference-1].append(neighbor_pos)
        # Only a difference greater then one can safe the bot
        return [*transitions_pos[1],*transitions_pos[2],*transitions_pos[3]]


class bomb_place_direction_finder(direction_finder):
    def __init__(self, logger):
        super().__init__(logger)
        self.default_action = np.zeros(7)
        self.last_target = None

    def preprocessing(self, field, pos, targets_pos, num):
        possible_targets_pos = targets_pos
        targets_pos = []
        add_last_target = True
        for possible_target_pos in possible_targets_pos:
            if self.is_placing_bomb_safe(field, possible_target_pos):
                targets_pos.append(possible_target_pos)
            if tuple(possible_target_pos) == self.last_target:
                add_last_target = False
        if self.last_target is not None:
            # Add the target of the last step to avoid losing the target between steps
            if add_last_target and self.is_placing_bomb_safe(field, self.last_target):
                targets_pos.append(self.last_target)
        return targets_pos

    def get_direction_of_shortest_path(self, field: np.array, pos: tuple, targets_pos: np.array, num: int, bomb_values:bomb_place_values):
        targets_pos = bomb_values.get_next_value_pos(num)
        if targets_pos is None:
            return self.default_action
        return super().get_direction_of_shortest_path(field, pos, targets_pos, num, bomb_values)

    def adjust_path_len(self, path, path_len, bomb_values:bomb_place_values):
        path_len = super().adjust_path_len(path, path_len)
        if path_len != 1000:
            path_len = - distance_value_adjustment(bomb_values.pre_dist_value_field[path[-1]],path_len-1)
        return path_len

    def get_direction_from_path(self, path, path_len, field, *args):
        self.last_target = path[-1]
        if len(path) == 1:
            action = self.default_action.copy()
            action[4] = 1
            action[-1] = path_len
            action[-2] = 1
            self.last_recomendation = "BOMB"
            self.last_target = None
            return action
        else:
            return super().get_direction_from_path(path, path_len, field, *args)

    def check_paths_for_action(self, field, pos, num, paths, path_lens, bomb_values):
        no_path_available, action = super().check_paths_for_action(field, pos, num, paths, path_lens, bomb_values)
        if no_path_available:
            action = self.get_direction_of_shortest_path(field, pos, np.array([None]), num, bomb_values)
            return True, action
        else:
            return no_path_available, action

    def is_placing_bomb_safe(self, field, bomb_pos):
        obstacle_field = np.where(field == 1, 0, 1)
        x, y = bomb_pos
        if x == 0 or x == 16 or y == 0 or y == 16:
            return False
        for j in range(1,4):
            if obstacle_field[x+j,y] == 0:
                if obstacle_field[x+j,y+1] == 0: return True
                if obstacle_field[x+j,y-1] == 0: return True
            else:
                break
        for j in range(1,4):
            if obstacle_field[x-j,y] == 0:
                if obstacle_field[x-j,y+1] == 0: return True
                if obstacle_field[x-j,y-1] == 0: return True
            else:
                break
        for j in range(1,4):
            if obstacle_field[x,y+j] == 0:
                if obstacle_field[x+1,y+j] == 0: return True
                if obstacle_field[x-1,y+j] == 0: return True
            else:
                break
        for j in range(1,4):
            if obstacle_field[x,y-j] == 0:
                if obstacle_field[x+1,y-j] == 0: return True
                if obstacle_field[x-1,y-j] == 0: return True
            else:
                break
        return False
