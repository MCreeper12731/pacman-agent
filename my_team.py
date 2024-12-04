import random
import util
import pickle
from capture_agents import CaptureAgent
from collections import deque
from game import Directions
from util import nearest_point
from network import *
import os

class QLearningAgent(CaptureAgent):
    """
    A Q-learning agent for Pacman Capture the Flag with proper transition observation.
    """

    def __init__(self, index, alpha=0.0001, gamma=0.9):
        super().__init__(index)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.count = 0

        self.last_game_state = None
        self.last_action = None  # Track the last action
        self.last_state_vector = None

        self.enable_load = 1
        
        self.q_network = NeuralNetwork(
            input_size=11*11*9 + 5 + 5 + 5,
            hidden_size=512,
            output_size=1
        )

        self.q_network_def = NeuralNetwork(
            input_size=11*11*9 + 5 + 5 + 5,
            hidden_size=512,
            output_size=1
        )

        self.load_weights() 

    def load_weights(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        weights_path_off = os.path.join(script_dir, 'weights_off')
        weights_path_def = os.path.join(script_dir, 'weights_def')

        if os.path.exists(weights_path_off) and self.enable_load == 1:
            with open(weights_path_off, 'rb') as f:
                loaded_data = pickle.load(f)
            w_in, w_out, b_h, b_out = loaded_data

            self.q_network.weights_input_hidden = w_in
            self.q_network.bias_hidden = b_h
            self.q_network.weights_hidden_output = w_out
            self.q_network.bias_output = b_out

        if os.path.exists(weights_path_def) and self.enable_load == 1:
            with open(weights_path_def, 'rb') as f:
                loaded_data = pickle.load(f)
            w_in, w_out, b_h, b_out = loaded_data

            self.q_network_def.weights_input_hidden = w_in
            self.q_network_def.bias_hidden = b_h
            self.q_network_def.weights_hidden_output = w_out
            self.q_network_def.bias_output = b_out

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.last_state_vector = None
        self.last_game_state = None
        self.last_action = None

    def extract_region(self, game_state, region_size=11, pad_value='-'):
        s = str(self.get_current_observation())
        s = [list(line) for line in s.splitlines()]
        s = s[:-1]
        posi = game_state.get_agent_position(0)

        rows = len(s)
        cols = len(s[0]) if rows > 0 else 0
        half_region = region_size // 2  # Calculate half the size for centering
        result = []

        center_col = posi[0]
        center_row = rows - posi[1] - 1

        # Iterate over the subregion rows
        for i in range(center_row - half_region, center_row + half_region + 1):
            row = []
            for j in range(center_col - half_region, center_col + half_region + 1):
                if 0 <= i < rows and 0 <= j < cols:  # Check if indices are within bounds
                    if s[i][j] == '.':
                        row.append(' ')
                    else:
                        row.append(s[i][j])
                else:
                    row.append(pad_value)  # Pad with the specified value if out of bounds
            result.append(row)

        return result

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def food_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        values = [self.get_features(game_state, a)['distance_to_food'] for a in actions]

        max_value = min(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        return random.choice(best_actions)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        sub = self.extract_region(game_state)
        should_return = int((game_state.get_agent_state(self.index).num_carrying/20)*4)

        state_vector = self.preprocess_state(sub, self.food_action(game_state), should_return)

        """
        Action options
        """
        # Bring back food 
        if game_state.get_agent_state(self.index).num_carrying > 8 and random.random() < 0.5:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            action = best_action
            #print("heading home!")

        # Explore as a baseline_agent move
        elif random.random() < self.epsilon or (self.offensive and not game_state.get_agent_state(self.index).is_pacman):
            values = [self.evaluate(game_state, a) for a in actions]

            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]

            food_left = len(self.get_food(game_state).as_list())

            if self.offensive and food_left <= 2:
                best_dist = 9999
                best_action = None
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    pos2 = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(self.start, pos2)
                    if dist < best_dist:
                        best_action = action
                        best_dist = dist
                action = best_action
            else:
                action = random.choice(best_actions)

        # Exploitation: Choose the action with the highest Q-value
        else:
            q_values = []
            for action in actions:
                # Create an input vector combining state and action
                action_map = {"North": [1, 0, 0, 0, 0], "South": [0, 1, 0, 0, 0],
                      "East": [0, 0, 1, 0, 0], "West": [0, 0, 0, 1, 0], "Stop": [0, 0, 0, 0, 1]}
                action_vector = action_map[action]
                input_vector = state_vector + action_vector

                if self.offensive:
                    q_value = self.q_network.forward(input_vector)[0]
                else:
                    q_value = self.q_network_def.forward(input_vector)[0]

                q_values.append((q_value, action))

            # Find the action with the maximum Q-value
            max_q_value = max(q_values, key=lambda x: x[0])[0]
            #print(max_q_value)
            #print(q_values)
            best_actions = [a for q, a in q_values if q == max_q_value]
            action = random.choice(best_actions)

        # Update the last state and action
        self.last_state_vector = state_vector 
        self.last_game_state = game_state
        self.last_action = action

        return action

    def preprocess_state(self, matrix, direction, should_return):
        # Flatten the 11x11 matrix and map characters to integers
        char_map = {' ': 0, '%': 1, '>': 2, '^': 3, '<': 4, 'v': 5, 'G': 6, 'o': 7, '-': 8}  # Example mapping
        num_chars = len(char_map)  # Number of unique characters

        # Flatten the matrix and one-hot encode each character
        matrix_flat = []
        for row in matrix:
            for char in row:
                # Create a one-hot vector for each character
                one_hot_vector = [0] * num_chars
                one_hot_vector[char_map[char]] = 1
                matrix_flat.extend(one_hot_vector)

        # One-hot encode the direction
        direction_map = {"North": [1, 0, 0, 0, 0], "South": [0, 1, 0, 0, 0],
                      "East": [0, 0, 1, 0, 0], "West": [0, 0, 0, 1, 0], "Stop": [0, 0, 0, 0, 1]}
        direction_encoded = direction_map[direction]

        # Normalize the should_return integer (assuming max value of 4)
        should_return_vector = [0] * 5
        should_return_vector[should_return] = 1

        # Combine into a single vector
        combined = matrix_flat + direction_encoded + should_return_vector
        return combined

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveQ(QLearningAgent):
    def __init__(self, index):
        super().__init__(index)
        self.offensive = 1
        self.enable_learn = 1
        self.epsilon = 0.4  # Exploration probability
        #self.enable_learn = 0
        #self.epsilon = 1.0

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        food_list = self.get_food(successor).as_list()

        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Compute distance to the nearest food
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        enemies = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
       
        if len(enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies]
            for d in dists:
                if d <= 5:
                    features['enemy_distance'] += d

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'enemy_distance': 1}


class DefensiveQ(QLearningAgent):
    def __init__(self, index):
        super().__init__(index)
        self.offensive = 0
        #self.enable_learn = 1
        #self.epsilon = 0.8  # Exploration probability
        self.enable_learn = 0
        self.epsilon = 1.0  # Exploration probability

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food_you_are_defending(successor).as_list()

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'distance_to_food': -5}


def create_team(first_index, second_index, is_red,
                first='OffensiveQ', second='DefensiveQ', num_training=0):
    """
    Create a team of Q-learning agents.
    """
    return [eval(first)(first_index), eval(second)(second_index)]
