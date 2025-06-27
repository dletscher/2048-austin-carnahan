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
        # Fix #2: Moderate learning rate
        self._learningRate = 0.1  # Reduced from 0.05
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
            
    # Fix #3: Take max value across rotations instead of sum
    def value(self, board):
        # Look at all rotations but take the maximum value
        values = []
        for turns in range(4):
            g = board.rotate(turns)
            values.append(self._valueTable.get(tuple(g._board[:8]), self._defaultValue))
        return max(values)  # Use maximum instead of sum
        
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
                
        # Store action values for diagnostics (we'll return this for training)
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
        batch_updates = []          # Track TD update magnitudes
        batch_action_counts = {'U': 0, 'D': 0, 'L': 0, 'R': 0}  # Track action distributions
        batch_action_values = []    # Track value differences between actions
        batch_empty_tiles = []      # Track number of empty tiles
        batch_game_length = []      # Track moves per game
        
        for trial in range(repetitions):
            total_games += 1
            batch_count += 1
            
            state = Game2048()
            state.randomize()
            
            # Per-game tracking
            moves_this_game = 0
            game_updates = []
            
            while not state.gameOver():
                self._startTime = time.time()
                
                # Track empty tiles
                empty_tiles = sum(1 for cell in state._board if cell == 0)
                batch_empty_tiles.append(empty_tiles)
                
                # Get action values and make move
                action_values = self.findMove(state)
                move = self.getMove()
                if not move:
                    break
                    
                # Track action distribution
                if move in batch_action_counts:
                    batch_action_counts[move] += 1
                
                # Track action value differences if we have multiple options
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
                
                for turns in range(4):
                    rotated_old = oldState.rotate(turns)
                    rotated_new = state.rotate(turns)

                    key = tuple(rotated_old._board[:8])
                    target = reward + self._discountFactor * self.value(rotated_new)
                    prediction = self._valueTable.get(key, self._defaultValue)
                    delta = self._learningRate * (target - prediction)

                    self._valueTable[key] = prediction + delta
                    game_updates.append(abs(delta))  # ← track actual delta
            
            # Track stats for this game
            final_score = state.getScore()
            max_tile = max(state._board) if state._board else 0
            
            # Update batch tracking
            batch_scores.append(final_score)
            batch_max_tile = max(batch_max_tile, max_tile)
            batch_max_score = max(batch_max_score, final_score)
            batch_game_length.append(moves_this_game)
            batch_updates.extend(game_updates)  # Add this game's updates
            
            # Print batch statistics every 100 games
            if batch_count == 100 or trial == repetitions - 1:
                if batch_scores:  # Make sure there are games in this batch
                    avg_score = sum(batch_scores) / len(batch_scores)
                    
                    # Convert max tile from exponent to actual value (2^max_tile)
                    actual_max_tile = 2 ** batch_max_tile
                    
                    # Calculate diagnostic metrics
                    avg_update = sum(batch_updates) / len(batch_updates) if batch_updates else 0
                    avg_game_length = sum(batch_game_length) / len(batch_game_length) if batch_game_length else 0
                    avg_empty_tiles = sum(batch_empty_tiles) / len(batch_empty_tiles) if batch_empty_tiles else 0
                    
                    # Calculate action distribution percentages
                    total_actions = sum(batch_action_counts.values())
                    action_pcts = {}
                    if total_actions > 0:
                        for action, count in batch_action_counts.items():
                            action_pcts[action] = (count / total_actions) * 100
                    
                    # Print metrics
                    print(f"Batch: {total_games // 100}, Games: {total_games}, " 
                          f"Avg Score: {avg_score:.1f}, Max Score: {batch_max_score}, "
                          f"Max Tile: {actual_max_tile:.0f}, Epsilon: {self._epsilon}")
                    
                    print(f"  Updates: {avg_update:.6f}, Moves/Game: {avg_game_length:.1f}, "
                          f"Empty Tiles: {avg_empty_tiles:.1f}")
                    
                    print(f"  Actions: U:{action_pcts.get('U', 0):.1f}% D:{action_pcts.get('D', 0):.1f}% "
                          f"L:{action_pcts.get('L', 0):.1f}% R:{action_pcts.get('R', 0):.1f}%")
                    
                    # Print value function statistics (min, avg, max)
                    values = list(self._valueTable.values())
                    if values:
                        print(f"  Value Range: Min={min(values):.3f}, Avg={sum(values)/len(values):.3f}, "
                              f"Max={max(values):.3f}, States={len(values)}")
                    
                # Reset batch tracking
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
