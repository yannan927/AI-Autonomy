from template import Agent
from Sequence.sequence_model import BOARD, COORDS, SequenceGameRule
from Sequence.sequence_utils import *
import random, time, copy


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

        square_count = 0

        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
            chip_str = ''.join([chips[r][c] for r, c in coord_list])
            for chip_chr in chip_str:
                if chip_chr == clr or chip_chr == sclr:
                    square_count +=1
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
        return (num_seq, square_count)


    def SelectAction(self, actions, game_state):
        start_time = time.time()
        action = actions[0]
        hand = game_state.agents[self.id].hand
        draft = game_state.board.draft
        chips = game_state.board.chips
        colour = game_state.agents[self.id].colour
        opp_colour = game_state.agents[self.id].opp_colour

        Best_draft = draft[0]
        Best_value = 0
        Second_draft = draft[0]
        Second_value = 0

        single_J = ''
        double_J = ''

    #Use the second chips to record all my hands except 'J' to see which of the card I'll capture fits best with my current game situation
        copy_chips = copy.deepcopy(chips)
        for card in hand:
            if not card == 'js' and not card == 'jh' and not card == 'jc' and not card == 'jd':
                coords = COORDS[card]
                for x,y in coords:
                    if copy_chips[x][y] == EMPTY:
                        copy_chips[x][y] = colour #place my hand in two position

        for card in draft:
            if card =='jc' or card == 'jd':
                double_J = card
                break # We will definitely take double_J
            if card == 'js' or card == 'jh':
                single_J = card

        if double_J:
            Best_draft = double_J
        elif single_J:
            Best_draft = single_J
        else: #Draw a regular card which fits best with my current game situation
            for card in draft:
                coords = COORDS[card]
                for x,y in coords:
                    if copy_chips[x][y] == EMPTY:
                        copy_chips[x][y] = colour
                        seqNum, chipNum = self.checkSeq(copy_chips, game_state.agents[self.id],(x,y))
                        copy_chips[x][y] = EMPTY
                        if seqNum > Best_value:
                            Best_value = seqNum
                            Best_draft = card
                        elif chipNum > Second_value: #if seqNum = 0, see if chipNum has changed
                            Second_value = chipNum
                            Second_draft = card
            if Best_value == 0:
                Best_draft = Second_draft

        if game_state.agents[self.id].trade:
            for action in actions:
                if action['draft_card'] == Best_draft:
                    return action
        else:
            cards = hand + [Best_draft]
            for i in range(len(cards)):
                if i =='jc' or i == 'jd':
                    card = cards.pop(i) # place front
                    cards.insert(0,card)
            for i in range(len(cards)):
                if i == 'js' or i == 'jh':
                    card = cards.pop(i)  # place front
                    cards.insert(0, card)
            #BFS
            myQueue = self.Queue()
            start_Node = (chips, cards, 0, [])
            myQueue.push(start_Node)
            First_Path = []
            Second_Path = []

            while not myQueue.isEmpty():
                node = myQueue.pop()
                chips, cards, seqNum, path = node
                if seqNum >= 2: #end
                    First_Path = path
                    break
                if seqNum == 1 and not Second_Path:
                    Second_Path = path
                for index in range(len(cards)):
                    card = cards[index]
                    Best_J_position = ()
                    Second_J_position = ()
                    Best_J_seqNum = 0
                    Best_J_count = 0
                    if card =='jc' or card == 'jd':
                        coords = COORDS[card]
                        for x, y in coords:
                            if chips[x][y] == EMPTY:
                                chips[x][y] = colour
                                tempNum,chipNum = self.checkSeq(chips, game_state.agents[self.id], (x,y))
                                chips[x][y] = EMPTY
                                if tempNum > Best_J_seqNum:
                                    Best_J_seqNum = tempNum
                                    Best_J_position = (x,y)
                                elif chipNum > Best_J_count:
                                    Best_J_count = chipNum
                                    Second_J_position = (x,y)
                        seqNum += Best_J_seqNum
                        temp_cards = cards[index+1:]
                        temp_chips = copy.deepcopy(chips)
                        if Best_J_position:
                            x,y = Best_J_position
                        else:
                            x,y = Second_J_position
                        temp_chips[x][y] = colour
                        myQueue.push((temp_chips, temp_cards, seqNum, path+[(card,(x,y))]))

                    elif card == 'js' or card == 'jh':
                        remove_coords = game_state.board.plr_coords[opp_colour]
                        if not remove_coords: #There is nothing to remove then skip
                            continue
                        for coord in remove_coords:
                            x,y = coord
                            if chips[x][y] == opp_colour: #See if the opponent occupies my sequence
                                chips[x][y] = colour
                                tempNum, chipNum = self.checkSeq(chips, game_state.agents[self.id], (x, y))
                                chips[x][y] = opp_colour
                                if tempNum > Best_J_seqNum:
                                    Best_J_seqNum = tempNum
                                    Best_J_position = (x, y)
                                elif chipNum > Best_J_count:
                                    Best_J_count = chipNum
                                    Second_J_position = (x, y)

                        temp_cards = cards[index + 1:]
                        temp_chips = copy.deepcopy(chips)
                        if Best_J_position:
                            x, y = Best_J_position
                        else:
                            x, y = Second_J_position
                        temp_chips[x][y] = EMPTY
                        myQueue.push((temp_chips, temp_cards, seqNum, path + [(card, (x,y))]))

                    else:
                        coords = COORDS[card]
                        for coord in coords:
                            x,y = coord
                            if chips[x][y] == EMPTY:
                                temp_chips = copy.deepcopy(chips)
                                temp_chips[x][y] = colour
                                tempNum, chipNum = self.checkSeq(temp_chips, game_state.agents[self.id], (x,y))
                                seqNum += tempNum
                                temp_cards = cards[index+1:]
                                myQueue.push((temp_chips,temp_cards,seqNum, path+[(card, (x,y))]))

            if First_Path:
                First_Play = ''
                First_Coord = ()
                for (card,coord) in First_Path:
                    if card in hand and not First_Play:
                        First_Play = card
                        First_Coord = coord
                        for action in actions:
                            if action['play_card'] == First_Play and action['draft_card'] == Best_draft and action['coords'] == First_Coord:
                                return action
            if not First_Path:
                First_Path = Second_Path

            Best_Action = ()
            Best_square_count = 0
            for action in actions:
                if action['type'] == 'place':
                    temp_colour = copy_chips[x][y]
                    copy_chips[x][y] = colour
                    seqNum, chipNum = self.checkSeq(copy_chips,game_state.agents[self.id], (x,y))
                    copy_chips[x][y] = temp_colour
                    if chipNum > Best_square_count:
                        Best_square_count = chipNum
                        Best_Action = action
            if Best_Action:
                return Best_Action
            else:
                return random.choice(actions)













