import random
import util
import pickle
from capture_agents import CaptureAgent
from collections import deque
from game import Directions
from util import nearest_point
import os

class DecisionPolicyAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.count = 0
        self.stagnation_counter = 0
        self.target_food_index = 0 
        self.position_history = []
        self.position_history_length = 5
        self.relocation_countdown = 0

        self.last_game_state = None
        self.last_action = None  # Track the last action

        self.score_diff_sensitivity = 3
        self.react_distance = 5


        # TODO: with noisy distance try to keep away
                
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.last_state_vector = None
        self.last_game_state = None
        self.last_action = None

        self.prevFoodDist = None
        self.prevEnemyDist = None
        self.prevCapsuleDist = None

        caves, entries = self.get_caves(game_state)
        self.cave_entry = entries
        self.caves = caves

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

    def get_caves(self, game_state):
        entries = [
                (27.0, 9.0),
                (27.0, 7.0),
                (26.0, 13.0),
                (24.0, 13.0),
                (20.0, 12.0),
                (21.0, 9.0),
                (22.0, 9.0),
                (19.0, 6.0),
                (25.0, 5.0),
                (25.0, 3.0),
                (25.0, 2.0),
                (23.0, 2.0),
                (21.0, 2.0),
                (17.0, 7.0)
                ]
        caves = [
                (21.0, 1.0),
                (23.0, 1.0),
                (25.0, 1.0),
                (26.0, 1.0),
                (26.0, 3.0),
                (23.0, 3.0),
                (21.0, 4.0),
                (22.0, 4.0),
                (23.0, 4.0),
                (24.0, 7.0),
                (23.0, 6.0),
                (24.0, 6.0),
                (25.0, 6.0),
                (17.0, 6.0),
                (20.0, 6.0),
                (21.0, 6.0),
                (21.0, 8.0),
                (22.0, 10.0),
                (21.0, 12.0),
                (23.0, 12.0),
                (24.0, 12.0),
                (26.0, 14.0),
                (28.0, 7.0),
                (28.0, 14.0),
                (28.0, 13.0),
                (28.0, 12.0),
                (28.0, 11.0),
                (28.0, 10.0),
                (28.0, 9.0)
                ]

        x0 = 31
        y0 = 15
        caves += [(x0 - x, y0 - y) for x, y in caves]
        entries += [(x0 - x, y0 - y) for x, y in entries]

        return (caves, entries)


    def compute_prev(self, game_state):
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        enemy_distance = -1
        food_distance = -1
        capsule_distance = -1
       
        if len(enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies]
            for d in dists:
                if d <= 5:
                    if enemy_distance == -1 or d < enemy_distance:
                        enemy_distance = d

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = game_state.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            food_distance = min_distance

        if len(capsules) > 0:  # This should always be True,  but better safe than sorry
            my_pos = game_state.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            capsule_distance = min_distance

        self.prevFoodDist = food_distance
        self.prevEnemyDist = enemy_distance
        self.prevCapsuleDist = capsule_distance

        self.position_history.append(my_pos)
        if(len(self.position_history) > 5):
            self.position_history.pop(0)

    def opponent_noisy_distances(self, game_state):
        if game_state.is_on_red_team(self.index):
            ally_indices = game_state.get_red_team_indices()
        else:
            ally_indices = game_state.get_blue_team_indices()

        all_noisy_distances = self.get_current_observation().get_agent_distances()

        my_pos = game_state.get_agent_state(self.index).get_position()

        for ally in ally_indices:
            all_noisy_distances[ally] = -1
        
        for dist in all_noisy_distances:
            if dist == -1:
                all_noisy_distances.remove(dist)

        return all_noisy_distances

    def is_on_start_pos(self, game_state):
            curr_agent = game_state.get_agent_state(self.index)
            curr_pos = curr_agent.get_position()
            start_pos = curr_agent.start.pos
            if int(curr_pos[0]) == start_pos[0] and int(curr_pos[1]) == start_pos[1]:
                return True
            return False

    def update_relocation(self, game_state):
        # TODO: fix
        curr_agent = game_state.get_agent_state(self.index)
        curr_pos = curr_agent.get_position()
        start_pos = curr_agent.start.pos

        if self.offensive:
            if curr_pos in self.position_history:
                self.stagnation_counter += 1
            else:
                self.target_food_index = 0
                self.stagnation_counter = 0

            if self.stagnation_counter >= 5:
                self.stagnation_counter = 0

                if self.target_food_index < len(self.get_food(game_state).as_list())-1 and not self.is_on_start_pos(game_state):
                    self.target_food_index += 1
                    self.relocation_countdown = 10

            if self.relocation_countdown-1 >= 0:
                #print("relocation")
                self.relocation_countdown -= 1

    def choose_action(self, game_state):
        # TODO: act on capsule
        # TODO: add cave entry for opposite start
        self.update_relocation(game_state) 
        self.compute_prev(game_state)

        """
        Action options
        """
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        action = random.choice(best_actions)

        self.last_game_state = game_state
        self.last_action = action


        return action

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        res = features * weights
        #print(action, res)
        return res


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

class OffensivePolicy(DecisionPolicyAgent):
    def __init__(self, index):
        super().__init__(index)
        self.offensive = 1

    def get_features(self, game_state, action):
        """
            seek food (15 * p)
            score = enemy_score - our_score

            d = [0-1] = min(1)(min_ghost_dist)/distance_to_start_acting)
            p = d^n
            diff_intensity = 3
            n = [0.1, 1] = 1 - max(0)min(0.9)((score/diff_intensity )* 0.9)

            close to ghosts -10
            
            avoid towards capsule: 1

            if in any danger (enemies != 0):
                if in cave:
                    go to entry of cave
                else:
                    don't go into cave

            TODO: get approx distance
            
            

        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)

        ate = len(self.get_food(game_state).as_list())-len(food_list)  # self.get_score(successor)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        enemies = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        enemy_distance = -1
        if len(enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies]
            for d in dists:
                if d <= 5:
                    if enemy_distance == -1 or d < enemy_distance:
                        enemy_distance = d

        features['enemy_distance'] = enemy_distance


        food_sorted_by_distance = sorted(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

        if len(food_list) > 0:
            min_distance = self.get_maze_distance(my_pos, food_sorted_by_distance[self.target_food_index])
            features['distance_to_food'] = min_distance
        
        if len(capsules) > 0:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            features['capsule_distance'] = min_distance

        if self.prevFoodDist != None:
            features['distance_to_food'] -= self.prevFoodDist

        if ate:
            features['distance_to_food'] = -1

        if self.prevEnemyDist != None:
            features['enemy_distance'] -= self.prevEnemyDist

        # TODO: ate?
        if self.prevCapsuleDist != None:
            features['capsule_distance'] -= self.prevCapsuleDist

        # Caves
        features['distance_to_entry'] = 0
        features['avoid_cave'] = 0

        current_pos = game_state.get_agent_state(self.index).get_position()

        # TODO: add noisy distance
        if enemy_distance < self.react_distance and enemy_distance != -1:
            if current_pos in self.caves:
                print("escaping cave")
                min_distance = min([self.get_maze_distance(my_pos, entry) for entry in self.cave_entry])
                features['distance_to_entry'] = min_distance
            else:
                if my_pos in self.caves:
                    #print("avoiding caves")
                    features['avoid_cave'] = 1


        # Bring back food 
        features['home_distance'] = self.get_maze_distance(self.start, my_pos)
    
        return features

    def get_weights(self, game_state, action):

        score = -self.get_score(game_state)
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        enemies = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        min_ghost_distance = 5
       
        if len(enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies]
            for d in dists:
                if d <= 5:
                    if min_ghost_distance == -1 or d < min_ghost_distance:
                        min_ghost_distance = d

        d = max(min((min_ghost_distance/self.react_distance), 1), 0)
        n = 1 - max(min((score/self.score_diff_sensitivity)*0.9, 0.9), 0)
        p = d**n

        if self.relocation_countdown > 0:
            p = -p 

        """
        # TODO: switch goal
        if game_state.get_agent_state(self.index).num_carrying > 8:
            self.offensive = 0
        """
        # TODO: maybe this already fixes stagnation
        go_home = 0
        if d < 1:
            p = 0
            go_home = -5

        return {'home_distance': go_home, 'distance_to_food': -10*p, 'enemy_distance': 10, 'capsule_distance': -1, 'distance_to_entry': -100, 'avoid_cave': -100}

class DefensivePolicy(DecisionPolicyAgent):
    def __init__(self, index):
        super().__init__(index)
        self.offensive = 0

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
                first='OffensivePolicy', second='DefensivePolicy', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
