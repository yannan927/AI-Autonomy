from template import Agent
from collections import defaultdict
import random
import copy
import math

RED     = 'r'
BLU     = 'b'
RED_SEQ = 'X'
BLU_SEQ = 'O'
JOKER   = '#'
EMPTY   = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3

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

COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

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

# new deck
deck_cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
         ['d', 'c', 'h', 's']]
deck_cards = deck_cards * 2  # Sequence uses 2 decks.


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.deck_cards = copy.deepcopy(deck_cards)
        random.shuffle(self.deck_cards)
        self.agentHandCards = [[] for _ in range(4)]
        self.agentActionIndex = [0, 0, 0, 0]
        self.discountFactor = 0.5

    def SelectAction(self, actions, game_state):
        # self.getCards(game_state)
        try:
            action = self.find_best_action(actions, game_state)
        except:
            random.choice(action)
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

    def find_best_action(self, actions, game_state):

        team_score = game_state.agents[self.id].score + game_state.agents[(self.id+2)%4].score
        opp_score  = game_state.agents[(self.id+1)%4].score + game_state.agents[(self.id-1)%4].score

        clr,sclr   = game_state.agents[self.id].colour, game_state.agents[self.id].seq_colour
        oc,os      = game_state.agents[self.id].opp_colour, game_state.agents[self.id].opp_seq_colour
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


    def coords_weight(self, chips, plr_state, coords, team_score, opp_score):
        clr,sclr   = plr_state.colour, plr_state.seq_colour
        oc,os      = plr_state.opp_colour, plr_state.opp_seq_colour
        lr,lc      = coords
        sc = 'x'
        chips[lr][lc] = sc #speical color
        weight = 0
        # consider opp_cards
        # opp_cards = self.agentHandCards[(self.id+1)%4] + self.agentHandCards[(self.id-1)%4]

        #All joker spaces become player chips for the purposes of sequence checking.
        for r,c in COORDS['jk']:
            chips[r][c] = clr

        #Heart of board strategy
        coord_list = [(4,4),(4,5),(5,4),(5,5)]
        if coords in coord_list:
            heart_chips = [chips[y][x] for x,y in coord_list]
            if heart_chips.count(clr) + heart_chips.count(sclr) == 3:
                weight += math.inf
            if (sclr in heart_chips or clr in heart_chips) and (oc in heart_chips or os in heart_chips):
                weight += 5
            if heart_chips.count(oc) + heart_chips.count(os) == 2:
                weight += 20
            if heart_chips.count(oc) + heart_chips.count(os) == 3:
                weight += 100
            else:
                weight += 10

        #Search vertical, horizontal, and both diagonals.
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
            chip_str   = ''.join([chips[r][c] for r,c in seq])

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
                        w2 += 1
                    elif pattern.count(EMPTY) == 2 and (sc in pattern) and not (
                            clr in pattern or sclr in pattern) and (pattern.count(os) <= 1):
                        w3 += 1
            if ws >= 2:
                weight = math.inf
            elif team_score > 0 and ws == 1:
                weight = math.inf
            elif opp_score > 0 and wo >= 1:
                weight = 100
            else:
                weight += 10*(ws + wo + w2*self.discountFactor + w3*(self.discountFactor**2))

        chips[lr][lc] = EMPTY
        for r,c in COORDS['jk']:
            chips[r][c] = JOKER #Joker spaces reset after sequence checking.
        return weight


