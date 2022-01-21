from copy import deepcopy
from collections import defaultdict
import math
import random
import time
from template import Agent


# Constant

BOARD = [['jk', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'jk'],
         ['6c', '5c', '4c', '3c', '2c', 'ah', 'kh', 'qh', 'th', 'ts'],
         ['7c', 'as', '2d', '3d', '4d', '5d', '6d', '7d', '9h', 'qs'],
         ['8c', 'ks', '6c', '5c', '4c', '3c', '2c', '8d', '8h', 'ks'],
         ['9c', 'qs', '7c', '6h', '5h', '4h', 'ah', '9d', '7h', 'as'],
         ['tc', 'ts', '8c', '7h', '2h', '3h', 'kh', 'td', '6h', '2d'],
         ['qc', '9s', '9c', '8h', '9h', 'th', 'qh', 'qd', '5h', '3d'],
         ['kc', '8s', 'tc', 'qc', 'kc', 'ac', 'ad', 'kd', '4h', '4d'],
         ['ac', '7s', '6s', '5s', '4s', '3s', '2s', '2h', '3h', '5d'],
         ['jk', 'ad', 'kd', 'qd', 'td', '9d', '8d', '7d', '6d', 'jk']]

# Store dict of cards and their coordinates for fast lookup.
COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row, col))

RED     = 'r'
BLU     = 'b'
RED_SEQ = 'X'
BLU_SEQ = 'O'
JOKER   = '#'
EMPTY   = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3

# new deck
deck_cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
         ['d', 'c', 'h', 's']]
deck_cards = deck_cards * 2  # Sequence uses 2 decks.

# board weight for initial state
alpha = 1
BOARD_WEIGHTS = alpha * [[0, 2, 1, 1, 1, 1, 1, 1, 2, 0],
                         [2, 3, 2, 2, 2, 2, 2, 2, 3, 2],
                         [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
                         [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
                         [1, 2, 3, 4, 0, 0, 4, 3, 2, 1],
                         [1, 2, 3, 4, 0, 0, 4, 3, 2, 1],
                         [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
                         [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
                         [2, 3, 2, 2, 2, 2, 2, 2, 3, 2],
                         [0, 2, 1, 1, 1, 1, 1, 1, 2, 0]]


class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        self.ucb = UCB(0.5)
        self.rule = RULE(self.id)
        self.deck_cards = deepcopy(deck_cards)
        random.shuffle(self.deck_cards)
        self.agentHandCards = [[] for _ in range(4)]
        self.agentActionIndex = [0, 0, 0, 0]

    def SelectAction(self, actions, game_state):
        self.getCards(game_state)
        if self.agentActionIndex[0] <= 3:
            action = self.rule.find_best_action(actions, game_state)
            return action
        else:
            if actions[0]['type'] == 'trade':
                action = self.rule.find_best_action(actions, game_state)
            else:
                coords = []
                for action in actions:
                    if action['type'] == 'place':
                        if (action['play_card'], action['coords']) not in coords:
                            coords.append((action['play_card'], action['coords']))
                mcts = MonteCarloTreeSearch(self.id, game_state, self.agentHandCards, coords, 0.9, self.ucb, 0.9)
                a = mcts.FindNextMove(AverageQfunc)
                real_action = [action for action in actions if action['draft_card'] == a[2] and action['coords'] == a[1]]
                action = random.choice(real_action)
        return action


    def getCards(self, game_state):
        for i in range(4):
            for action_reward in game_state.agents[i].agent_trace.action_reward[self.agentActionIndex[i]:]:
                play_card = action_reward[0]['play_card']
                draft_card = action_reward[0]['draft_card']
                if play_card in self.agentHandCards[i]:
                    self.agentHandCards[i].remove(play_card)
                else:
                    self.deck_cards.remove(play_card)
                self.agentHandCards[i].append(draft_card)
                self.deck_cards.remove(draft_card)
                self.agentActionIndex[i] += 1
        if not game_state.agents[self.id].last_action:
            for hand in game_state.agents[self.id].hand:
                self.agentHandCards[self.id].append(hand)
                self.deck_cards.remove(hand)

class State:

    def __init__(self, agent_id, board, hands, agents, last_action):
        self.agent_id = agent_id
        self.board = board
        self.hands = hands
        self.agents = agents
        self.last_action = last_action
        self.visited_count = 0
        self.total_reward = [0, 0, 0, 0]

    def getActions(self):

        # random_card_num = 6 - len(self.hands[self.agent_id])
        # random_cards = random.choices(deck_cards, k=random_card_num)
        if self.hands[self.agent_id]:
            actions = []
            hand_cards = self.hands[self.agent_id]
            hand_cards = set(hand_cards)
            for card in hand_cards:
                if card == 'jc' or card == 'jd':
                    self.hands[self.agent_id].remove(card)
                elif card == 'jh' or card == 'js':
                    self.hands[self.agent_id].remove(card)
                else:
                    num = 0
                    for coord in COORDS[card]:
                        if coord in self.board.empty_coords and ((card, coord) not in actions):
                            actions.append((card, coord))
                        else:
                            num += 1
                    if num == 2:
                        # remove dead card
                        self.hands[self.agent_id].remove(card)
            # actions = [('_', coord) for coord in self.board.empty_coords]
            return actions
        else:
            return None

    # TODO: check and give weight choose for select action
    def checkSeq(self):
        pass

    def has_neighbor(self, coord):
        clr = self.agents[self.agent_id].colour
        oc = self.agents[self.agent_id].opp_colour
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (coord[0] + i, coord[1]+j) in self.board.plr_coords[clr]:
                    return True
                if (coord[0] + i, coord[1]+j) in self.board.plr_coords[oc]:
                    return True
        return False

    def actionFilter(self, actions):
        if not actions:
            return None
        if len(actions) > 4:
            action_weight = []
            chips = self.board.chips
            plr_state = self.agents[self.agent_id]
            team_score = plr_state.score + self.agents[(self.agent_id+2) % 4].score
            opp_score = self.agents[(self.agent_id+1) % 4].score + self.agents[(self.agent_id+3) % 4].score
            for action in actions:
                coord = action[1]
                r, c = coord
                weight = RULE.coords_weight(chips, plr_state, coord, team_score, opp_score)
                weight += BOARD_WEIGHTS[r][c]
                if weight > 1:
                    action_weight.append((action, weight))
            action_weight.sort(key=lambda tup: tup[1], reverse=True)
            return [action[0] for action in action_weight[0:4]]
        else:
            return actions

    def execute(self, board, hands, action):
        clr = self.agents[self.agent_id].colour
        r, c = action[1]
        # execute one move
        board.chips[r][c] = self.agents[self.agent_id].colour
        board.plr_coords[clr].append(action[1])
        board.empty_coords.remove(action[1])
        hands[self.agent_id].remove(action[0])
        # for action in
        # self.hands.remove()
        # move to next agent
        # self.agent_id = (self.agent_id+1) % 4
        return board, hands, action

    def getReward(self):
        chips = self.board.chips
        clr, sclr = self.agents[self.agent_id].colour, self.agents[self.agent_id].seq_colour
        oc, os = self.agents[self.agent_id].opp_colour, self.agents[self.agent_id].opp_seq_colour

        num_seq = 0
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = self.last_action[1]

        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            num_seq += 10

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq in [vr, hz, d1, d2]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                num_seq += 10
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        sequence_len = 0
                    if sequence_len == 4:
                        num_seq += 1
                    if sequence_len >= 5:
                        num_seq += 4
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            num_seq += 5
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        if num_seq > 0:
            found = True
        reward = [0, 0, 0, 0]
        if found:
            reward[self.agent_id] = num_seq
            reward[(self.agent_id + 1) % 4] = -num_seq
            reward[(self.agent_id + 2) % 4] = num_seq
            reward[(self.agent_id + 3) % 4] = -num_seq
        if len(self.board.empty_coords) <= 24:
            found = True
        return found, reward

    def nextAgent(self):
        self.agent_id = (self.agent_id + 1) % 4

    def isTerminal(self):
        if not self.hands[self.agent_id]:
            return True
        chips = self.board.chips
        pre_agent_id = (self.agent_id-1) % 4
        clr, sclr = self.agents[pre_agent_id].colour, self.agents[pre_agent_id].seq_colour
        oc, os = self.agents[pre_agent_id].opp_colour, self.agents[pre_agent_id].opp_seq_colour
        lr, lc = self.last_action[1]
        num_seq = 0
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x

        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            num_seq += 10

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq in [vr, hz, d1, d2]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                num_seq += 10
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        sequence_len = 0
                    if sequence_len == 4:
                        num_seq += 1
                    if sequence_len >= 5:
                        num_seq += 4
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            num_seq += 5
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        if num_seq > 0:
            found = True
        if len(self.board.empty_coords) <= 24:
            found = True
        return found

    def chooseAction(self, actions):
        # weighted based action choose
        if len(actions) >= 1:
            action_weight = {}
            chips = self.board.chips
            plr_state = self.agents[self.agent_id]
            team_score = plr_state.score + self.agents[(self.agent_id+2) % 4].score
            opp_score = self.agents[(self.agent_id+1) % 4].score + self.agents[(self.agent_id+3) % 4].score
            for action in actions:
                coord = action[1]
                r, c = coord
                weight = RULE.coords_weight(chips, plr_state, coord, team_score, opp_score)
                action_weight[action] = weight + BOARD_WEIGHTS[r][c]
            action = max(action_weight, key=action_weight.get)
            return action
        else:
            return actions

    def chooseDraft(self):
        card_weight = {}
        chips = self.board.chips
        plr_state = self.agents[self.agent_id]
        team_score = plr_state.score + self.agents[(self.agent_id + 2) % 4].score
        opp_score = self.agents[(self.agent_id + 1) % 4].score + self.agents[(self.agent_id + 3) % 4].score
        for card in self.board.draft:
            weight = 0
            if card == 'jc' or card == 'jd':
                return card
            elif card == 'jh' or card == 'js':
                return card
            else:
                for r, c in COORDS[card]:
                    if EMPTY == self.board.chips[r][c]:
                        coord = (r, c)
                        weight_ = RULE.coords_weight(chips, plr_state, coord, team_score, opp_score)
                        weight_ += BOARD_WEIGHTS[r][c]
                        if weight_ > weight:
                            weight = weight_
            card_weight[card] = weight
        card = max(card_weight, key=card_weight.get)
        return card


class Node:

    def __init__(self, state, parent):

        self.children = []
        self.state = state
        self.parent = parent

    def expand(self, actions=None):
        """Expand all the possible children of the node
        Generate all the possible children of the node, the next node belong to the opponent,
        We do self play to train ourselves
        """
        # Get all move based on the current game state
        next_agent_id = (self.state.agent_id + 1) % 4
        draft_card = None
        if actions is None:
            actions = self.state.getActions()
        else:
            draft_card = self.state.chooseDraft()
        actions = self.state.actionFilter(actions)
        if not actions:
            print(';')
        for action in actions:
            # Get the next board: deep copy the current board and hand cards.
            next_board = deepcopy(self.state.board)
            next_hands = deepcopy(self.state.hands)
            next_board, next_hands, last_action = self.state.execute(next_board, next_hands, action)
            if draft_card:
                next_hands[(self.state.agent_id-1) % 4] += draft_card
                last_action = (*last_action, draft_card)
            # Add the new node to its children
            self.children.append(Node(State(next_agent_id, next_board, next_hands,
                                            self.state.agents, last_action), self))


class MonteCarloTreeSearch:

    def __init__(self, agent_id, game_state, hands, actions, time_limit, ucb, discount_factor):
        self.actions = actions
        self.player_id = agent_id
        self.time_limit = time_limit
        self.ucb = ucb
        self.discount_factor = discount_factor
        # Initial tree
        self.root = Node(State(agent_id, game_state.board, hands, game_state.agents, None), None)
        # Here we expand the root directly to prevent empty selection
        self.root.expand(actions)

    def FindNextMove(self, q_func):

        begin_time = time.time()
        while time.time() - begin_time < self.time_limit:
            # Select the leaf node with the higher UCB1 value
            expand_node = self.Selection(self.root, q_func)
            # Expand the node, only expand if it is visited more than one times before.
            # This encourage breath search one more time instead of expanding nodes.
            # If the node is the end of the game, use the expand_node to simulate.
            child = expand_node
            # A higher visited_count would lead to more simulation. Not enough computing power provided!
            if expand_node.state.visited_count > 0 and not expand_node.state.isTerminal():
                children = self.Expansion(expand_node)
                # It may not have children. Also, for the first time, it would not expand its children.
                if len(children) > 0:
                    child = self.Choose(children)
            # Simulation
            rewards, move_count = self.Simulation(child)
            # Back propagation, backup both our reward and our opponent's reward (Self training)
            self.BackPropagate(child, rewards, move_count, self.discount_factor)
        # Find the best move with the highest win score
        # Choose the move that can bring the max Q value
        best_child = self.bestChild(self.root.children, q_func)
        return best_child.state.last_action

    def Selection(self, root, q_func):
        """ Select the leaf node with the highest UCB to expand"""
        node = root
        while len(node.children) > 0:
            node = self.ucb.FindBestChildNode(node, q_func)
        return node

    @staticmethod
    def Expansion(node):
        """ Expand the node, add children into its leaves """
        node.expand()
        return node.children

    @staticmethod
    def Choose(children):
        return random.choice(children)

    @staticmethod
    def Simulation(child):
        """ Use some simple and costless strategy to simulate """
        # Deep copy first to avoid change the game_state in the node
        board_copy = deepcopy(child.state.board)
        hands_copy = deepcopy(child.state.hands)
        gs_copy = State(child.state.agent_id, board_copy, hands_copy, child.state.agents, child.state.last_action)
        found, reward = gs_copy.getReward()
        move_count = 0
        while not found:
            # Update move
            # if not any(hands_copy):
            #     break
            actions = gs_copy.getActions()
            if not actions:
                # gs_copy.nextAgent()
                break
            # TODO: provide different strategy for the simulation
            # Strategy 1 -- random select:
            action = random.choice(actions)
            # Strategy 2 -- choose best action:
            # action = gs_copy.chooseAction(actions)
            gs_copy.execute(gs_copy.board, gs_copy.hands, action)
            found, reward = gs_copy.getReward()
            gs_copy.nextAgent()
            move_count += 1
        return reward, move_count


    @staticmethod
    def BackPropagate(node, rewards, move_count, discount_factor):
        """ Back propagation """
        real_rewards = [reward * (discount_factor ** move_count) for reward in rewards]
        p_node = node
        while p_node is not None:
            # Increase the visited count
            p_node.state.visited_count += 1
            p_node.state.total_reward[0] += real_rewards[0]
            p_node.state.total_reward[1] += real_rewards[1]
            p_node.state.total_reward[2] += real_rewards[2]
            p_node.state.total_reward[3] += real_rewards[3]
            p_node = p_node.parent

    def bestChild(self, children, q_func):
        return max(children, key=q_func)

        # if len(best_child) == 1:
        #     return best_child.state.last_action
        # else:
        #     actions = [child.state.last_action for child in best_child]
        #     action_weight = {}
        #     chips = self.root.state.board.chips
        #     plr_state = self.root.state.agents[self.root.state.agent_id]
        #     team_score = plr_state.score + self.root.state.agents[(self.root.state.agent_id + 2) % 4].score
        #     opp_score = self.root.state.agents[(self.root.state.agent_id + 1) % 4].score \
        #                 + self.root.state.agents[(self.root.state.agent_id + 3) % 4].score
        #     for action in actions:
        #         coord = action[1]
        #         weight = RULE.coords_weight(chips, plr_state, coord, team_score, opp_score)
        #         action_weight[action] = weight
        #     action = max(action_weight, key=action_weight.get)
        #     return action

class UCB:
    """
    Epsilon-Greedy multi-armed bandit
    Attributes
        exploration_constant : explore probability
    """

    def __init__(self, exploration_constant):

        self.exploration_constant = exploration_constant

    def FindBestChildNode(self, node, q_func):
        """Find the best node through comparing the highest UCB
        Args:
            node: expand_node
            q_func: how to calculate the q value, the child node would be its parameter.
        Returns:
            The node with the max UCB
        """
        parent_visited_count = node.state.visited_count
        max_ucb = float('-inf')
        best_child = None
        # Child state belong to the opponent, we need to find one with the best win_scores_sum of ourselves
        for child in node.children:
            if child.state.visited_count == 0:
                return child
            else:
                tmp = self._CalculateUCB(parent_visited_count, q_func(child), child.state.visited_count)
                if tmp > max_ucb:
                    max_ucb = tmp
                    best_child = child
        return best_child

    def _CalculateUCB(self, total_visit, q_value, node_visited_count):
        """Calculate UCB
        Args:
            total_visit: the visit count of the parent node
            q_value: the Q value of the node
            node_visited_count: the visit count of the current node
        Returns:
            The UCB value
        """
        if node_visited_count == 0:
            return float('inf')
        return q_value + self.exploration_constant * math.sqrt(math.log(total_visit) / node_visited_count)


def AverageQfunc(node):
    if node.state.visited_count == 0:
        return 0
    parent_id = (node.state.agent_id - 1) % 4
    return node.state.total_reward[parent_id] / node.state.visited_count


def AggressiveQfunc(node):
    if node.state.visited_count == 0:
        return 0
    parent_id = (node.state.agent_id - 1) % 4
    child_id = (node.state.agent_id + 1) % 4
    return (node.state.total_reward[parent_id] - node.state.total_reward[child_id]) / node.state.visited_count


class RULE:

    def __init__(self, agent_id):
        self.id = agent_id

    def find_best_action(self, actions, game_state):

        team_score = game_state.agents[self.id].score + game_state.agents[(self.id+2)%4].score
        opp_score  = game_state.agents[(self.id+1)%4].score + game_state.agents[(self.id-1)%4].score

        # clr,sclr   = game_state.agents[self.id].colour, game_state.agents[self.id].seq_colour
        # oc,os      = game_state.agents[self.id].opp_colour, game_state.agents[self.id].opp_seq_colour
        draft_cards = game_state.board.draft
        coords_list = []
        coords_weight = {}
        trade_flag = False
        remove_flag = False # consider remove later
        remove_list = []
        for action in actions:
            if action['type'] == 'trade':
                trade_flag = True
            if action['type'] != 'remove':
                if action['coords'] not in coords_list:
                    coords_list.append(action['coords'])
            else:
                if action['coords'] not in remove_list:
                    remove_list.append(action['coords'])
                    remove_flag = True
        if remove_flag:  # remove policy
            # only remove heart card
            coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
            heart_chips = [game_state.board.chips[y][x] for x,y in coord_list]
            oc = game_state.agents[self.id].opp_colour
            if heart_chips.count(oc) >= 1:
                coords = random.choice([coord for coord in remove_list if coord in coord_list])
                r, c = coords
                game_state.board.chips[r][c] = EMPTY  # sequence bug
                draft = self.draft_weight(game_state, draft_cards, team_score, opp_score)
                action = [action for action in actions if action['coords']==coords and action['draft_card']==draft]
                return action[0]
        if trade_flag:  # trade policy
            draft = self.draft_weight(game_state, draft_cards, team_score, opp_score)
            action = [action for action in actions if action['draft_card']==draft]
            return action[0]

        # normal policy
        for coords in coords_list:
            r, c = coords
            weight = self.coords_weight(game_state.board.chips, game_state.agents[self.id], coords, team_score, opp_score)
            coords_weight[coords] = weight + BOARD_WEIGHTS[r][c]
        coords = max(coords_weight, key=coords_weight.get)
        r, c = coords
        game_state.board.chips[r][c] = game_state.agents[self.id].colour   # sequence bug
        # draft weight
        draft = self.draft_weight(game_state, draft_cards, team_score, opp_score)
        action = [action for action in actions if action['coords']==coords and action['draft_card']==draft]

        return action[0]

    # Select best draft card from all draft cards
    def draft_weight(self, game_state, draft_cards, team_score, opp_score):
        draft_weight = {}
        for draft in draft_cards:
            weight = 0
            if draft[0] == 'j':
                weight += 1000
            else:
                for r,c in COORDS[draft]:
                    if EMPTY == game_state.board.chips[r][c]:
                        coords = (r,c)
                        weight_ = self.coords_weight(game_state.board.chips, game_state.agents[self.id], coords, team_score, opp_score)
                        weight_ += BOARD_WEIGHTS[r][c]
                        if weight_ > weight:
                            weight = weight_
            draft_weight[draft] = weight
        draft = max(draft_weight, key=draft_weight.get)
        return draft

    @staticmethod
    def two_eyed(empty_coords, chips, plr_state):
        coords = {}
        for coord in empty_coords:
            seq = RULE.seqCheck(chips, plr_state, coord)
            if seq > 0:
                coords[coord] = seq
        if len(coords) >= 1:
            coord = max(coords, key=coords.get)
            return coord
        else:
            return None
    @staticmethod
    def seqCheck(chips, plr_state, last_coords):
        clr, sclr = plr_state.colour, plr_state.seq_colour
        oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords
        num_seq = 0
        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
                oc in heart_chips or os in heart_chips):
            num_seq += 2

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
        for seq in [vr, hz, d1, d2]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                num_seq += 2

            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        sequence_len = 0
                    if sequence_len >= 5:
                        num_seq += 1
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            num_seq += 1
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        return num_seq

    @staticmethod
    def coords_weight(chips, plr_state, last_coord, team_score, opp_score):
        clr, sclr = plr_state.colour, plr_state.seq_colour
        oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
        lr, lc = last_coord
        sc = 'x'
        chips[lr][lc] = sc  # speical color
        weight = 0
        # consider opp_cards
        # opp_cards = self.agentHandCards[(self.id+1)%4] + self.agentHandCards[(self.id-1)%4]

        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # Heart of board strategy
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if last_coord in coord_list:
            heart_chips = [chips[y][x] for x, y in coord_list]
            if heart_chips.count(clr) + heart_chips.count(sclr) == 3:
                weight += math.inf
            if (sclr in heart_chips or clr in heart_chips) and (oc in heart_chips or os in heart_chips):
                weight += 3
            if heart_chips.count(oc) + heart_chips.count(os) == 2:
                weight += 20
            if heart_chips.count(oc) + heart_chips.count(os) == 3:
                weight += 50
            else:
                weight += 10

        # Search vertical, horizontal, and both diagonals.
        vr, hz, d1, d2 = [], [], [], []
        for i in range(10):
            vr.append((lr, i))
            hz.append((i, lc))
        # d1
        for i in range(10 - max(lr, lc)):
            d1.append((lr + i, lc + i))
        for i in range(1, min(lr, lc) + 1):
            d1.append((lr - i, lc - i))
        # d2
        for i in range(min(lr + 1, 10 - lc)):
            d2.append((lr - i, lc + i))
        for i in range(1, min(10 - lr, lc + 1)):
            d2.append((lr + i, lc - i))
        coord_list = [vr, hz]
        if len(d1) >= 5:
            coord_list.append(d1)
        if len(d2) >= 5:
            coord_list.append(d2)

        ws, wo, w2, w3 = 0, 0, 0, 0  # w1 for five in a line, w2 for four, w3 for three
        for seq in coord_list:
            chip_str = ''.join([chips[r][c] for r, c in seq])

            for start_idx in range(5):
                pattern = chip_str[start_idx:start_idx + 5]
                if len(pattern) == 5:
                    # self
                    if EMPTY not in pattern and (sc in pattern) and not (oc in pattern or os in pattern) and (
                            pattern.count(sclr) <= 1):
                        ws += 1
                    elif pattern.count(EMPTY) == 1 and (sc in pattern) and not (
                            oc in pattern or os in pattern) and (pattern.count(sclr) <= 1):
                        w2 += 1
                    elif pattern.count(EMPTY) == 2 and (sc in pattern) and not (
                            oc in pattern or os in pattern) and (pattern.count(sclr) <= 1):
                        w3 += 1
                    # opp
                    if EMPTY not in pattern and (sc in pattern) and not (clr in pattern or sclr in pattern) and (
                            pattern.count(os) <= 1):
                        wo += 1
                    elif pattern.count(EMPTY) == 1 and (sc in pattern) and not (
                            clr in pattern or sclr in pattern) and (pattern.count(os) <= 1):
                        w3 += 1

            if ws >= 2:
                weight = math.inf
            elif team_score >= 1 and ws == 1:
                weight = math.inf
            elif opp_score >= 1 and wo >= 1:
                weight = 100
            else:
                weight += 20 * (ws + wo + w2 * 0.5 + w3 * (0.5 ** 2))

        chips[lr][lc] = EMPTY
        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.
        return weight