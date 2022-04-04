from typing import List
import torch as T
import numpy as np

import events as e
from .callbacks import state_to_features
from .model import Agent

TRANSITION_HISTORY_SIZE = 3  
RECORD_ENEMY_TRANSITIONS = 1.0  
REVERSE_ACTON = "REVERSE_ACTON"
CHOSE_RECOMMENDED = "CHOSE_RECOMMENDED"
DID_NOT_CHOSE_RECOMMENDED = "DID_NOT_CHOSE_RECOMMENDED"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up training")
    self.high_score = -1000000
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    if (not self.load_model):
        epsilon = 1.0
    else:
        epsilon = 0.3
    self.agent.init_training(gamma=0.99, epsilon=epsilon, batch_size=64, eps_end=0.2, max_mem_size=100000, eps_dec=5e-6, target_sync=500) #5e-7
    self.scores, self.average_scores, self.eps_history = [], [], []
    self.n_max = 2000000
    self.n_save = 100
    self.n = 0
    self.score = 0
    self.last_action = 10


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state == None:
        return

    action = np.argwhere(np.array(ACTIONS) == self_action)[0][0]
    features = self.act_state

    for event in events:
        if event == e.MOVED_UP:
            self.action = 0
        elif event == e.MOVED_RIGHT:
            self.action = 1
        elif event == e.MOVED_DOWN:
            self.action = 2
        elif event == e.MOVED_LEFT:
            self.action = 3
        else:
            self.action = 10
    
        
    if np.abs(self.action - self.last_action) == 2:
        events.append(REVERSE_ACTON)
    
    self.last_action = self.action

    self.logger.debug(features)
    if was_recommended(self, action):
        events.append(CHOSE_RECOMMENDED)
    else:
        events.append(DID_NOT_CHOSE_RECOMMENDED)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    self.score += reward

    self.agent.store_transition(state=features, action=action, reward=reward, state_=state_to_features(self, new_game_state), done=False)
    self.agent.learn()



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    action = np.argwhere(np.array(ACTIONS) == last_action)[0][0]

    for event in events:
        if event == e.MOVED_UP:
            self.action = 0
        elif event == e.MOVED_RIGHT:
            self.action = 1
        elif event == e.MOVED_DOWN:
            self.action = 2
        elif event == e.MOVED_LEFT:
            self.action = 3
        else:
            self.action = 10
   

    if np.abs(self.action - self.last_action) == 2:
        events.append(REVERSE_ACTON)
    
    self.last_action = self.action

    features = state_to_features(self, last_game_state)
    if was_recommended(self, action):
        events.append(CHOSE_RECOMMENDED)
    else:
        events.append(DID_NOT_CHOSE_RECOMMENDED)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    self.score += reward
    self.agent.store_transition(state=features, action=action, reward=reward, state_=features, done=True)
    self.agent.learn()

    self.scores.append(self.score)
    self.eps_history.append(self.agent.epsilon)
    
    if self.n % self.n_save == 0 or self.n == self.n_max:
        self.average_scores.append(np.mean(self.scores[-100:]))
        if self.high_score < self.average_scores[-1]:
            self.high_score = self.average_scores[-1]
            np.save("scores_best.npy",np.array(self.scores))
            np.save("epsilons_best.npy",np.array(self.eps_history))
            np.save("average_scores_best.npy", np.array(self.average_scores))
            self.agent.save_agent(self.path[:-3] + "_best.pt")
        np.save("scores.npy",np.array(self.scores))
        np.save("epsilons.npy",np.array(self.eps_history))
        np.save("average_scores.npy", np.array(self.average_scores))
        self.agent.save_agent(self.path)


    self.n += 1
    self.score = 0

def was_recommended(self, action):
    for finder in self.finders:
        recom = finder.last_recomendation
        if action == 0 and recom == "UP":
            return True
        elif action == 1 and recom == "RIGHT":
            return True
        elif action == 2 and recom == "DOWN":
            return True
        elif action == 3 and recom == "LEFT":
            return True
        elif action == 4 and recom == "WAIT":
            return True
        elif action == 5 and recom == "BOMB":
            return True
    return False

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 4,
        e.KILLED_OPPONENT: 20,
        e.BOMB_DROPPED: -0.5,
        e.CRATE_DESTROYED: 2,
        e.KILLED_SELF: -4,
        e.GOT_KILLED: -3,
        e.SURVIVED_ROUND: 0,
        e.WAITED: -0.6,
        e.INVALID_ACTION: -1,
        e.MOVED_UP: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_DOWN: -0.1,
        REVERSE_ACTON: -0.6,
        CHOSE_RECOMMENDED: 0.3,
        DID_NOT_CHOSE_RECOMMENDED: -0.7
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

