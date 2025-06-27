import pickle
import random
import numpy as np  # For some statistical calculations
from Game2048 import *

class Player(BasePlayer):
    def __init__(self, timeLimit):
        BasePlayer.__init__(self, timeLimit)
        # Initialize table
        self._valueTable = {}
        self._defaultValue = 0.5
        self._learningRate = .05
        self._discountFactor = .95
        self._epsilon = 0.1  # 10% random moves

    def loadData(self, filename):
        print('Loading data')
        try:
            # Check if file exists and has content
            import os
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                print(f"File {filename} does not exist or is empty. Using default initialization.")
                return
                
            with open(filename, 'rb') as dataFile:
                self._valueTable = pickle.load(dataFile)
                print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}. Using default initialization.")
            
    def saveData(self, filename):
        print('Saving data')
        with open(filename, 'wb') as dataFile:
            pickle.dump(self._valueTable, dataFile)
            
    def value(self, board):
        # The table stores the value of the first row.
        # Look at all rotations and add there values so we
        # also get the last row, first column and last column.
        v = 0.
        for turns in range(4):
            g = board.rotate(turns)
            v += self._valueTable.get(tuple(g._board[:8]), 0.0)
        return v
        
    def findMove(self, board):
        actions = board.actions()
        if not actions:
            self.setMove('')
            return
            
        # Epsilon-greedy: 10% random moves
        if random.random() < self._epsilon:
            randomMove = random.choice(list(actions))
            self.setMove(randomMove)
            return
            
        # Otherwise select best move
        bestValue = float('-inf')
        bestMove = ''
        action_values = {}  # Track values for each action for diagnostics
        
        for a in actions:
            # Finding the expected (or average) value of the state after the move is taken
            v = 0
            for (g, p) in board.possibleResults(a):
                v += p * self.value(g)
            
            action_values[a] = v
            
            if v > bestValue:
                bestValue = v
                bestMove = a

        self._last_action_values = action_values
        self.setMove(bestMove)
        return action_values
        
    def train(self, repetitions):
        # Initialize tracking variables
        batch_scores = []
        batch_max_tile = 0
        batch_max_score = 0
        batch_count = 0
        total_games = 0
        
        # Diagnostic metrics
        batch_updates = []        
        batch_action_counts = {'U': 0, 'D': 0, 'L': 0, 'R': 0} 
        batch_action_values = []  
        batch_empty_tiles = []    
        batch_game_length = []
        
        for trial in range(repetitions):
            total_games += 1
            batch_count += 1
            
            state = Game2048()
            state.randomize()

            moves_this_game = 0
            game_updates = []
            
            while not state.gameOver():
                self._startTime = time.time()
                
                empty_tiles = sum(1 for cell in state._board if cell == 0)
                batch_empty_tiles.append(empty_tiles)
                
                # Get action values and make move
                action_values = self.findMove(state)
                move = self.getMove()
                if not move:
                    break
                    
                if move in batch_action_counts:
                    batch_action_counts[move] += 1
                
                if action_values and len(action_values) > 1:
                    values = list(action_values.values())
                    value_range = max(values) - min(values)
                    batch_action_values.append(value_range)
                
                # Make the move
                oldState = state
                state, reward = state.result(move)
                moves_this_game += 1
                
                # Calculate and track update magnitude
                update = self._learningRate * (reward + self._discountFactor*self.value(state) - self.value(oldState))
                game_updates.append(abs(update))  # Track absolute value of updates
                
                # Update the table
                for turns in range(4):
                    rotated = oldState.rotate(turns)
                    state_key = tuple(rotated._board[:8])
                    # Make sure key exists before updating
                    if state_key not in self._valueTable:
                        self._valueTable[state_key] = 0.0
                    self._valueTable[state_key] += update
            
            final_score = state.getScore()
            max_tile = max(state._board) if state._board else 0
            
            batch_scores.append(final_score)
            batch_max_tile = max(batch_max_tile, max_tile)
            batch_max_score = max(batch_max_score, final_score)
            batch_game_length.append(moves_this_game)
            batch_updates.extend(game_updates)  # Add this game's updates
            
            # Print batch statistics every 100 games
            if batch_count == 100 or trial == repetitions - 1:
                if batch_scores:  
                    avg_score = sum(batch_scores) / len(batch_scores)
                    
                    actual_max_tile = 2 ** batch_max_tile
                    avg_update = sum(batch_updates) / len(batch_updates) if batch_updates else 0
                    avg_game_length = sum(batch_game_length) / len(batch_game_length) if batch_game_length else 0
                    avg_empty_tiles = sum(batch_empty_tiles) / len(batch_empty_tiles) if batch_empty_tiles else 0
                    total_actions = sum(batch_action_counts.values())
                    action_pcts = {}
                    if total_actions > 0:
                        for action, count in batch_action_counts.items():
                            action_pcts[action] = (count / total_actions) * 100
                    print(f"Batch: {total_games // 100}, Games: {total_games}, " 
                          f"Avg Score: {avg_score:.1f}, Max Score: {batch_max_score}, "
                          f"Max Tile: {actual_max_tile:.0f}, Epsilon: {self._epsilon}")
                    
                    print(f"  Updates: {avg_update:.6f}, Moves/Game: {avg_game_length:.1f}, "
                          f"Empty Tiles: {avg_empty_tiles:.1f}")
                    
                    print(f"  Actions: U:{action_pcts.get('U', 0):.1f}% D:{action_pcts.get('D', 0):.1f}% "
                          f"L:{action_pcts.get('L', 0):.1f}% R:{action_pcts.get('R', 0):.1f}%")

                    values = list(self._valueTable.values())
                    if values:
                        print(f"  Value Range: Min={min(values):.3f}, Avg={sum(values)/len(values):.3f}, "
                              f"Max={max(values):.3f}")
                    
                batch_scores = []
                batch_max_tile = 0
                batch_max_score = 0
                batch_count = 0
                batch_updates = []
                batch_action_counts = {'U': 0, 'D': 0, 'L': 0, 'R': 0}
                batch_action_values = []
                batch_empty_tiles = []
                batch_game_length = []
                    
if __name__ == '__main__':
    # Perform training
    a = Player(1)
    a.loadData('MyData')
    a.train(100000)
    a.saveData('MyData')
