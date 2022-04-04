from typing import List
import torch as T
import numpy as np

import events as e
from .callbacks import state_to_features


REVERSE_ACTON = "REVERSE_ACTON"
CHOSE_RECOMMENDED = "CHOSE_RECOMMENDED"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up training")
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    if (not self.load_model):
        epsilon = 1.0
    else:
        epsilon = 0.3
    self.agent.init_training(gamma=0.99, epsilon=epsilon, batch_size=64, eps_end=0.2, max_mem_size=100000, eps_dec=1e-6, target_sync=500) #5e-7
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
    state = self.act_state

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

    feature = state[1]
    self.logger.debug(feature)
    if was_recommended(action, feature):
        events.append(CHOSE_RECOMMENDED)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    self.score += reward

    self.agent.store_transition(state=state, action=action, reward=reward, state_=state_to_features(self, new_game_state), done=False)
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

    state = state_to_features(self, last_game_state)
    feature = state[1]
    if was_recommended(action, feature):
        events.append(CHOSE_RECOMMENDED)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    self.score += reward
    self.agent.store_transition(state=state, action=action, reward=reward, state_=state, done=True)
    self.agent.learn()

    self.scores.append(self.score)
    self.eps_history.append(self.agent.epsilon)
    
    if self.n % self.n_save == 0 or self.n == self.n_max:
        self.average_scores.append(np.mean(self.scores[-100:]))
        np.save("scores.npy",np.array(self.scores))
        np.save("epsilons.npy",np.array(self.eps_history))
        np.save("average_scores.npy", np.array(self.average_scores))
        self.agent.save_agent(self.path)


    self.n += 1
    self.score = 0

def was_recommended(action, feature):
    bool = False
    if action == 0 and (feature[0]==1 or feature[4]==1 or feature[8]==1 or feature[12]==1):
        bool = True
    elif action == 1 and (feature[1]==1 or feature[5]==1 or feature[9]==1 or feature[13]==1):
        bool = True
    elif action == 2 and (feature[2]==1 or feature[6]==1 or feature[10]==1 or feature[14]==1):
        bool = True
    elif action == 3 and (feature[3]==1 or feature[7]==1 or feature[11]==1 or feature[15]==1):
        bool = True
    elif action == 5 and feature[16] == 1:
        bool = True
    return bool

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
        e.CRATE_DESTROYED: 1,
        e.KILLED_SELF: -4,
        e.GOT_KILLED: -3,
        e.SURVIVED_ROUND: 0,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -1,
        e.MOVED_UP: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.MOVED_DOWN: -0.05,
        REVERSE_ACTON: -0.7,
        CHOSE_RECOMMENDED: 0.4,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

