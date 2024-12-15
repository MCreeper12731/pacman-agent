import random
import util
from capture_agents import CaptureAgent
from collections import deque
from game import Directions
from distance_calculator import Distancer
from util import nearest_point
import os

class DecisionPolicyAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.count = 0
        self.stagnation_counter = 0
        self.position_history = []
        self.position_history_length = 5

        self.last_game_state = None
        self.last_action = None  # Track the last action

        self.score_diff_sensitivity = 3
        self.react_distance = 5

        self.custom_distancer = None
        self.walls_layout = None
        self.cave_thr = 4



    ################
    # Initialization
    ################
                
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.last_state_vector = None
        self.last_game_state = None
        self.last_action = None

        self.prevFoodDist = None
        self.prevEnemyDist = None
        self.prevCapsuleDist = None
        self.prevNoisyDist = None
        self.centers = []

        self.prob_maps = [self.initialize_prob_map(game_state), self.initialize_prob_map(game_state)]

        self.initialize_walls_caves(game_state)
        
        self.initialize_caves(game_state)

    def initialize_walls_caves(self, game_state):
        layout = game_state.data.layout.walls
        self.walls_layout = layout.deep_copy()
        self.cave_layout = layout.deep_copy()

    def initialize_caves(self, game_state):
        n = game_state.data.layout.height
        m = game_state.data.layout.width

        num_refreshs = 3

        for i in range(num_refreshs):
            for y in range(1, n-1):
                for x in range(1, m-1):
                    if not self.is_wall(x, y):
                        self.cave_flood_fill(game_state, (x, y))

        for y in range(1, n-1):
                for x in range(1, m-1):
                    if self.is_single(x, y):
                        self.cave_layout[x][y] = 'E'

                    if self.is_entry(x, y):
                        if self.cave_layout[x+1][y] == False: 
                            self.cave_layout[x+1][y] = 'G'
                            continue
                        if self.cave_layout[x-1][y] == False: 
                            self.cave_layout[x-1][y] = 'G'
                            continue
                        if self.cave_layout[x][y+1] == False: 
                            self.cave_layout[x][y+1] = 'G'
                            continue
                        if self.cave_layout[x][y-1] == False: 
                            self.cave_layout[x][y-1] = 'G'
                            continue
                        self.cave_layout[x][y] = 'C'

    def initialize_prob_map(self, game_state):
        n = game_state.data.layout.height
        m = game_state.data.layout.width

        total_positions = n*m
        prob_map = {}
        for x in range(1, m-1):
            for y in range(1, n-1):
                prob_map[(x, y)] = 1 / total_positions

        return prob_map

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

    def set_target_food(self, game_state):
        food_list = self.get_food(game_state).as_list()

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        self.food_sorted_by_ratio = sorted(food_list, key=lambda food: 
                  (self.get_maze_distance(my_pos, food)/(max(1, min([self.get_maze_distance(food, center) for center in self.centers])) if len(self.centers) > 0 else 1)))



    def compute_prev(self, game_state):
        width = game_state.data.layout.width

        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        enemy_distance = -1
        food_distance = -1
        capsule_distance = -1
        noisy_distance = 0
       
        if len(enemies) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies]
            for d in dists:
                if d <= 5:
                    if enemy_distance == -1 or d < enemy_distance:
                        enemy_distance = d
        
        if len(food_list) > 0:
            min_distance = self.get_maze_distance(my_pos, self.food_sorted_by_ratio[0])
            food_distance = min_distance
        
        if len(capsules) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            capsule_distance = min_distance

        if self.offensive:
            if len(self.centers) > 0:
                noisy_distance = min([self.get_maze_distance(my_pos, center) for center in self.centers])
        else:
            centers_relevant = []
            for center in self.centers:
                if game_state.is_on_red_team(self.index):
                    if center[0] < width/2:
                        centers_relevant.append(center)
                else:
                    if center[0] > width/2:
                        centers_relevant.append(center)

            if len(centers_relevant) > 0:
                noisy_distance = min([self.get_maze_distance(my_pos, center) for center in centers_relevant])

        self.prevFoodDist = food_distance
        self.prevEnemyDist = enemy_distance
        self.prevCapsuleDist = capsule_distance
        self.prevNoisyDist = noisy_distance

        self.position_history.append(my_pos)
        if(len(self.position_history) > 5):
            self.position_history.pop(0)


    def is_on_start_pos(self, game_state):
            curr_agent = game_state.get_agent_state(self.index)
            curr_pos = curr_agent.get_position()
            start_pos = curr_agent.start.pos
            if int(curr_pos[0]) == start_pos[0] and int(curr_pos[1]) == start_pos[1]:
                return True
            return False

    ################
    #  Noisy distance
    ################

    def opponent_noisy_distances(self, game_state):
        if game_state.is_on_red_team(self.index):
            ally_indices = game_state.get_red_team_indices()
        else:
            ally_indices = game_state.get_blue_team_indices()

        all_noisy_distances = self.get_current_observation().get_agent_distances()

        for ally in ally_indices:
            if ally < len(all_noisy_distances):
                all_noisy_distances[ally] = -1
        
        for dist in all_noisy_distances:
            if dist == -1:
                all_noisy_distances.remove(dist)

        return all_noisy_distances

    def decay_probabilities(self, decay_factor=0.8):
        for ix in range(2):
            for pos in self.prob_maps[ix]:
                self.prob_maps[ix][pos] *= decay_factor
            # Normalize again to keep probabilities valid
            total_prob = sum(self.prob_maps[ix].values())
            for pos in self.prob_maps[ix]:
                self.prob_maps[ix][pos] /= total_prob



    def find_approximations(self, game_state, n=5):
        height = game_state.data.layout.height
        width = game_state.data.layout.width

        def calculate_center(top_left):
            xc, yc = top_left
            return (xc + n // 2, yc + n // 2)

        highest_sum = float("-inf")
        top_left = None
        
        centers = []
        for ix in range(2):
            for y in range(1, height - n):
                for x in range(1, width - n):
                    total = 0
                    for i in range(n):
                        for j in range(n):
                            if 0 <= x + i < width and 0 <= y + j < height:
                                total += self.prob_maps[ix][(x + i, y + j)]

                    square_sum = total 
                    center_int = calculate_center((x, y))

                    if square_sum > highest_sum and not self.is_wall(center_int[0], center_int[1]):
                        highest_sum, top_left = square_sum, (x, y)

            centers.append(calculate_center(top_left))

        return centers

    def update_prob_map(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        noisy_distances = self.opponent_noisy_distances(game_state)

        for ix in range(2):
            if ix >= len(noisy_distances):
                continue

            updated_map = {}

            for pos, prob in self.prob_maps[ix].items():
                if not self.walls_layout[pos[0]][pos[1]]:
                    dist = self.get_maze_distance(my_pos, pos)
                    likelihood = 1 / (1 + abs(dist - noisy_distances[ix]))  # Higher for closer matches
                    updated_map[pos] = prob * likelihood
                else:
                    updated_map[pos] = 0

            # Normalize probabilities to sum to 1
            total_prob = sum(updated_map.values())
            for pos in updated_map:
                updated_map[pos] /= total_prob

            self.prob_maps[ix] = updated_map

            ix += 1


    def display_prob_map(self, game_state):
        n = game_state.data.layout.height
        m = game_state.data.layout.width

        for y in range(n):
            row = []
            for x in range(m):
                value = self.prob_maps[0].get((x, y), 0)  # Default to 0 if the key is not in the map
                row.append(f"{value:.3f}")
            print(" ".join(row))
                


    def transition_probabilities(self):
        for ix in range(2):
            new_prob_map = {}

            for pos, prob in self.prob_maps[ix].items():
                x = pos[0]
                y = pos[1]

                neighbors = []
                neighbors.append(pos)
                if not self.walls_layout[x+1][y]: neighbors.append((x+1, y))
                if not self.walls_layout[x-1][y]: neighbors.append((x-1, y))
                if not self.walls_layout[x][y+1]: neighbors.append((x, y+1))
                if not self.walls_layout[x][y-1]: neighbors.append((x, y-1))

                spread_prob = prob / len(neighbors)
                for neighbor in neighbors:
                    if neighbor not in new_prob_map:
                        new_prob_map[neighbor] = 0
                    new_prob_map[neighbor] += spread_prob
            # Normalize
            total_prob = sum(new_prob_map.values())
            for pos in new_prob_map:
                new_prob_map[pos] /= total_prob

            self.prob_maps[ix] = new_prob_map

    ######################
    # Unified defense logic 
    ######################

    def is_potential_constraint(self, pos):
        x = pos[0]
        y = pos[1]
        if (self.walls_layout[x+1][y] and
            self.walls_layout[x-1][y] and
            not self.walls_layout[x][y+1] and
            not self.walls_layout[x][y-1]):
            return True

        if (not self.walls_layout[x+1][y] and
            not self.walls_layout[x-1][y] and
            self.walls_layout[x][y+1] and
            self.walls_layout[x][y-1]):
            return True
        return False 

    def reset_intercept(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if len(invaders) == 0:
            """
            if self.custom_distancer != None:
                print("reset")
            """
            self.custom_distancer = None
            self.layout_changed = None

    def should_intercept(self, game_state, action):
        successor = self.get_successor(game_state, action)

        if game_state.is_on_red_team(self.index):
            ally_indices = game_state.get_red_team_indices()
        else:
            ally_indices = game_state.get_blue_team_indices()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]


        other_ghost_pos = game_state.get_agent_state(ally_indices[0]).get_position()
        my_pos = game_state.get_agent_state(self.index).get_position()
        next_pos = successor.get_agent_state(self.index).get_position()

        if self.is_cave(my_pos[0], my_pos[1]):
            return False

        for invader in invaders:
            inv_pos = invader.get_position()
            if self.get_maze_distance(my_pos, inv_pos) > self.get_maze_distance(other_ghost_pos, inv_pos):
                if self.is_potential_constraint((int(next_pos[0]), int(next_pos[1]))):
                    if self.custom_distancer == None:
                        """
                        if self.index == 0:
                            print("Red", "should intercept", next_pos)
                        else:
                            print("orange", "should intercept", next_pos)
                        """

                        self.set_custom_distancer(game_state, next_pos)
                        return True
        return False

    def get_defense_features(self, game_state, action):
        if game_state.is_on_red_team(self.index):
            ally_indices = game_state.get_red_team_indices()
        else:
            ally_indices = game_state.get_blue_team_indices()

        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food_you_are_defending(successor).as_list()
        capsules = self.get_capsules_you_are_defending(game_state)

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.constraint_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        if len(capsules) > 0:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            features['capsule_distance'] = min_distance

        features['ally_distance'] = self.get_maze_distance(my_pos, game_state.get_agent_state(ally_indices[0]).get_position())

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.constraint_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        width = game_state.data.layout.width

        # Noisy distance
        centers_relevant = []
        for center in self.centers:
            if game_state.is_on_red_team(self.index):
                if center[0] < width/2:
                    centers_relevant.append(center)
            else:
                if center[0] > width/2:
                    centers_relevant.append(center)


            
        noisy_distance = 0
        if len(centers_relevant) > 0:
            noisy_distance = min([self.constraint_maze_distance(my_pos, center) for center in centers_relevant])

        features['approach_noisy'] = noisy_distance

        """
        if self.prevNoisyDist != None:
            features['approach_noisy'] -= self.prevNoisyDist
        """

        return features

    def get_defense_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)

        capsules = self.get_capsules(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        capsule_distance = 0
        if len(capsules) > 0:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            capsule_distance = min_distance

        on_capsule = 1
        if capsule_distance <= 1:
            on_capsule = 0

        if_ghost = 1
        if game_state.get_agent_state(self.index).is_pacman:
            if_ghost = 0
        
        return {'approach_noisy': -10*if_ghost, 'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -100, 'stop': -100, 'reverse': -2, 'distance_to_food': -5*0, 'capsule_distance': -10*on_capsule*0, 'ally_distance': 15*0}

    def constraint_maze_distance(self, pos1, pos2):
        if self.custom_distancer != None:
            if (self.layout_changed.walls[int(pos1[0])][int(pos1[1])] == True
            or self.layout_changed.walls[int(pos2[0])][int(pos2[1])] == True):
                return 100
            return self.custom_distancer.get_distance(pos1, pos2)
        return self.get_maze_distance(pos1, pos2)
    
    def set_custom_distancer(self, game_state, constraint):
        self.layout_changed = game_state.data.layout.deep_copy()
        self.layout_changed.walls[int(constraint[0])][int(constraint[1])] = True
        self.custom_distancer = Distancer(self.layout_changed)
        self.custom_distancer.get_maze_distances()
    
    ######################
    # Caves
    ######################

    def is_wall(self, x, y):
        return self.walls_layout[x][y]
    def is_entry(self, x, y):
        return self.cave_layout[int(x)][int(y)] == 'E'
    def is_single(self, x, y):
        return self.cave_layout[x][y] == 'S'
    def is_cave(self, x, y):
        return self.cave_layout[int(x)][int(y)] == 'C'
    def get_escapes(self, game_state):
        n = game_state.data.layout.height
        m = game_state.data.layout.width
        escapes = []
        for y in range(1, n-1):
            for x in range(1, m-1):
                if self.cave_layout[x][y] == 'G':
                    escapes.append((x, y))
        return escapes

    def is_potential_entry(self, pos):
        num_exits = 0
        directions = []

        if not self.walls_layout[pos[0]+1][pos[1]]: 
            num_exits += 1
            directions.append((pos[0]+1, pos[1]))
        if not self.walls_layout[pos[0]-1][pos[1]]: 
            num_exits += 1
            directions.append((pos[0]-1, pos[1]))
        if not self.walls_layout[pos[0]][pos[1]+1]: 
            num_exits += 1
            directions.append((pos[0], pos[1]+1))
        if not self.walls_layout[pos[0]][pos[1]-1]: 
            num_exits += 1
            directions.append((pos[0], pos[1]-1))

        if num_exits == 2 or num_exits == 1:
            return directions
        return None

    def cave_flood_fill(self, game_state, pos):
        directions = self.is_potential_entry(pos)
        n = game_state.data.layout.height
        m = game_state.data.layout.width

        if directions != None: 
            has_cave = 0
            for direction in directions:
                if len(directions) == 1:
                    self.cave_layout[pos[0]][pos[1]] = 'S'
                    return
                stack = [direction]
                region = []
                visited = [[False for _ in range(n)] for _ in range(m)]
                visited[pos[0]][pos[1]] = True
                while stack:
                    cx, cy = stack.pop()
                    if visited[cx][cy]:
                        continue
                    visited[cx][cy] = True
                    region.append((cx, cy))

                    if len(region) > self.cave_thr:
                        region = []
                        break

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if not self.is_wall(nx, ny) and not self.is_entry(nx, ny):
                            stack.append((nx, ny))

                for coor in region:
                    self.cave_layout[coor[0]][coor[1]] = 'C'
                if len(region) > 0:
                    has_cave = 1
                    self.cave_layout[pos[0]][pos[1]] = 'E'


    def choose_action(self, game_state):

        # 0 == Red
        self.set_target_food(game_state)
        """
        if self.index == 1:
            print(game_state.get_agent_state(self.index).num_carrying)
#            print("food:", self.food_sorted_by_ratio[0])
        """

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

        if self.custom_distancer == None and self.should_intercept(game_state, action):
            action = self.choose_action(game_state)
            #print("curr", game_state.get_agent_state(self.index).get_position(), "dir", action)

        self.reset_intercept(game_state)

        self.decay_probabilities()
        self.update_prob_map(game_state)
        self.transition_probabilities()

        #self.display_prob_map(game_state)

        self.centers = self.find_approximations(game_state)
        """
        if self.index == 1:
            print("pos:", game_state.get_agent_state(self.index).get_position())
            print("centers", self.centers)
        """

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
        score = self.get_score(game_state)

        if score >= 8 and not game_state.get_agent_state(self.index).is_pacman:
            #print("protect")
            self.offensive = 0

        features = util.Counter()
        successor = self.get_successor(game_state, action)

        ###################
        # Defensive offensive agent
        ##################
        if self.offensive == 0:
            return self.get_defense_features(game_state, action)

        ###################
        # Offensive offensive agent
        ##################
        if self.offensive == 1:
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

            if len(food_list) > 0:
                min_distance = self.get_maze_distance(my_pos, self.food_sorted_by_ratio[0])
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

            """
            if self.prevCapsuleDist != None:
                features['capsule_distance'] -= self.prevCapsuleDist
            """
            features['capsule_distance'] = 0

            # Caves
            features['distance_to_entry'] = 0
            features['avoid_cave'] = 0

            current_pos = game_state.get_agent_state(self.index).get_position()

            # Noisy distance
            noisy_distance = 0
            if len(self.centers) > 0:
                noisy_distance = min([self.get_maze_distance(my_pos, center) for center in self.centers])

            features['avoid_noisy'] = noisy_distance

            if self.prevNoisyDist != None:
                features['avoid_noisy'] -= self.prevNoisyDist

            if (enemy_distance < self.react_distance and enemy_distance != -1):
                if self.is_cave(current_pos[0], current_pos[1]):
                    print("escaping cave")
                    min_distance = min([self.get_maze_distance(my_pos, escape) for escape in self.get_escapes(game_state)])
                    features['distance_to_entry'] = min_distance
                else:
                    if self.is_cave(my_pos[0], my_pos[1]) or self.is_entry(my_pos[0], my_pos[1]):
                        print("avoiding caves")
                        features['avoid_cave'] = 1


            # Bring back food 
            features['home_distance'] = self.get_maze_distance(self.start, my_pos)

            # Retreating
            features['retreat'] = 0
            if game_state.get_agent_state(self.index).is_pacman and not my_state.is_pacman:
                #print("retreat")
                features['retreat'] = 1

            """
            if self.index == 1:
                print(action, features, self.prevFoodDist)
            """

    
        return features

    def get_weights(self, game_state, action):
        
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        ###################
        # Defensive offensive agent
        ##################
        if self.offensive == 0:
            return self.get_defense_weights(game_state, action)

        ###################
        # Offensive offensive agent
        ##################
        if self.offensive == 1:
            score = -self.get_score(game_state)
            
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

            go_home = 0
            if d < 1:
                p = 0
                go_home = -5

            if game_state.get_agent_state(self.index).num_carrying >= 8:
                #print("going home!")
                go_home = -100

            return {'home_distance': go_home, 'distance_to_food': -20*p, 'enemy_distance': 100, 'capsule_distance': -1, 'distance_to_entry': -100, 'avoid_cave': -100, 'avoid_noisy': 2, 'retreat':-2}

class DefensivePolicy(DecisionPolicyAgent):
    def __init__(self, index):
        super().__init__(index)
        self.offensive = 0

    def get_features(self, game_state, action):
        return self.get_defense_features(game_state, action)

    def get_weights(self, game_state, action):
        return self.get_defense_weights(game_state, action)

def create_team(first_index, second_index, is_red,
                first='OffensivePolicy', second='DefensivePolicy', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]
