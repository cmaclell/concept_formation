import json
import random
from trestle import Trestle

class TicTacToe:
    
    def __init__(self):
        self.board = {}
        self.board[0] = {}
        self.board[1] = {}
        self.board[2] = {}

        self.board[0][0] = " "
        self.board[0][1] = " "
        self.board[0][2] = " "
        self.board[1][0] = " "
        self.board[1][1] = " "
        self.board[1][2] = " "
        self.board[2][0] = " "
        self.board[2][1] = " "
        self.board[2][2] = " "

    def get_player(self):
        x_count = 0
        o_count = 0

        for row in self.board:
            for col in self.board[row]:
                if self.board[row][col] == 'X':
                    x_count += 1
                elif self.board[row][col] == 'O':
                    o_count += 1

        if x_count <= o_count:
            return "X"
        else:
            return "O"

    def value(self, x, y):
        return self.board[y][x]

    def validity(self, x, y):
        if x < 0 or x > 2:
            return False
        if y < 0 or y > 2:
            return False
        if self.board[y][x] != ' ':
            return False

        return True

    def move(self, x, y):
        assert self.board[y][x] == ' '
        player = self.get_player()
        self.board[y][x] = player

    def display(self):
        for row in range(2,-1,-1):
            print("".join([value for (key, value) in self.board[row].items()]))

    def game_state(self):
        state = {}

        for row in self.board:
            for col in self.board[row]:
                state[str(row) + '_' + str(col)] = {'x': row, 'y': col,
                                                    'piece': self.board[row][col]}

        state['row0'] = ['row', '0_0', '0_1', '0_2']
        state['row1'] = ['row', '1_0', '1_1', '1_2']
        state['row2'] = ['row', '2_0', '2_1', '2_2']
        state['col0'] = ['row', '0_0', '1_0', '2_0']
        state['col1'] = ['row', '0_1', '1_1', '2_1']
        state['col3'] = ['row', '0_2', '1_2', '2_2']
        state['diag0'] = ['row', '0_0', '1_1', '2_2']
        state['diag1'] = ['row', '0_2', '1_1', '2_0']
        
        winner = self.winner()
        if winner:
            state['winner'] = winner

        return state

    def winner(self):
        for row in range(3):
            if (self.board[row][0] == self.board[row][1] and
                self.board[row][1] == self.board[row][2] and
                self.board[row][0] != ' '):
                return self.board[row][0]

        for col in range(3):
            if (self.board[0][col] == self.board[1][col] and
                self.board[1][col] == self.board[2][col] and
                self.board[0][col] != ' '):
                return self.board[0][col]

        if (self.board[0][0] == self.board[1][1] and 
            self.board[1][1] == self.board[2][2] and
            self.board[0][0] != ' '):
            return self.board[0][0]

        if (self.board[0][2] == self.board[1][1] and
            self.board[1][1] == self.board[2][0] and
            self.board[0][2] != ' '):
            return self.board[0][2]

        tie = True
        for row in self.board:
            for col in self.board[row]:
                if self.board[row][col] == ' ':
                    tie = False

        if tie:
            return "TIE"
        
        return None

def get_user_move():
    while True:
        game.display()
        print("--%s's move--" % game.get_player())
        while True:
            x = int(input("Enter x value of move: "))
            if x < 0 or x > 2:
                print("invalid range, try again.")
            else:
                break

        while True:
            y = int(input("Enter y value of move: "))
            if y < 0 or y > 2:
                print("invalid range, try again.")
            else:
                break
    
        if game.value(x,y) != ' ':
            print("Spot already taken, try again.")
        else:
            break
    
    return x,y

def query_user(game, x, y):

    game.display() 
    val = None
    while val != 'y' and val != 'n' and val != '':
        val = input("Would %s, %s be a good move [y,n]? " % (x, y))

    return val == 'y' or val == ''

if __name__ == "__main__":

    model = Trestle()

    while True:
        game = TicTacToe()
        while game.winner() == None:
            #concept = model.trestle_categorize(game.game_state())
            with open('visualize/output.json', 'w') as f:
                f.write(json.dumps(model.output_json()))
            
            # human player - ai learns from their moves.
            x,y = get_user_move()

            state = game.game_state()
            state['player'] = game.get_player()
            state['x'] = x
            state['y'] = y
            state['valid'] = True
            game.move(x,y)
            model.ifit(state)


            # end if you need to
            winner = game.winner() 
            if winner == "TIE":
                print("TIE game")
                break
            elif winner == 'X' or winner == 'O':
                print("%s won!" % winner)
                break

            # AI player

            game.display()
            player = game.get_player()
            print()
            print("--%s's move (AI)--" % game.get_player())
            state = game.game_state()
            state['winner'] = player
            state['valid'] = True
            state['player'] = player
            
            concept = model.trestle_categorize(state)

            x = None
            y = None
            negative = set()
            while concept:
                if ('x' not in concept.av_counts or 
                    'y' not in concept.av_counts):
                    concept = concept.parent
                    continue

                options = []
                for xv in concept.av_counts['x']:
                    for yv in concept.av_counts['y']:
                        if (xv, yv) in negative:
                            continue
                        options += [(xv, yv)] * (concept.av_counts['x'][xv] *
                                               concept.av_counts['y'][yv])
                
                if not options:
                    if concept.parent:
                        concept = concept.parent
                        continue
                    else:
                        x = None
                        y = None
                        break
                
                x, y = random.choice(options)

                if game.validity(x, y) == False or (x,y) in negative or query_user(game, x, y) == False:
                    negative.add((x,y))
                    invalid_state = game.game_state()
                    invalid_state['valid'] = False
                    invalid_state['player'] = game.get_player()
                    invalid_state['x'] = x
                    invalid_state['y'] = y

                    model.ifit(invalid_state)
                    concept = model.trestle_categorize(state)
                    continue
                else:
                    break
                
            if x == None or y == None:
                print("I'm not sure what to do, can you show me?")
                x,y = get_user_move()
                state = game.game_state()
                state['player'] = game.get_player()
                state['x'] = x
                state['y'] = y
                state['valid'] = True
                game.move(x,y)
                model.ifit(state)
            else:
                print("AI moved %s, %s" % (x,y))
                state = game.game_state()
                state['player'] = game.get_player()
                state['x'] = x
                state['y'] = y
                state['valid'] = True
                game.move(x,y)
                model.ifit(state)

            # end if you need to
            winner = game.winner() 
            if winner == "TIE":
                print("TIE game")
                break
            elif winner == 'X' or winner == 'O':
                print("%s won!" % winner)
                break

            # for debugging
            #game.display()
            #break
