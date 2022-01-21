from template import Agent
from Sequence.sequence_model import BOARD, COORDS, SequenceGameRule

import random, time, copy

RED     = 'r'
BLU     = 'b'
RED_SEQ = 'X'
BLU_SEQ = 'O'
JOKER   = '#'
EMPTY   = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3


class myAgent(Agent):
    class Queue:
        def __init__(self):
            self.list = []

        def push(self, item):
            self.list.insert(0, item)

        def pop(self):
            return self.list.pop()

        def isEmpty(self):
    #return true if the queue is empty
            return len(self.list) == 0

        def getLen(self):
            return len(self.list)

    def __init__(self, _id):
        super().__init__(_id)

    def checkSeq(self, chips, plr_state, last_coords):
        clr, sclr = plr_state.colour, plr_state.seq_colour
        oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type = TRADSEQ
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords

        # All joker spaces become player chips for the purposes of sequence checking.
        for r, c in COORDS['jk']:
            chips[r][c] = clr

        # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb'] += 2
            seq_coords.append(coord_list)

        # Search vertical, horizontal, and both diagonals.
        vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]

        chip_count = 0

        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            for chip_chr in chip_str:
                if chip_chr == clr or chip_chr == sclr:
                    chip_count +=1
            # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name] += 2
                seq_coords.append(coord_list)
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i + 1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        break
            else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                                clr + sclr + clr * 3, sclr + clr * 4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            seq_found[seq_name] += 1
                            seq_coords.append(coord_list[start_idx:start_idx + 5])
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return (num_seq, chip_count)

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