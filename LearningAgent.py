import pickle
import random
from Game2048 import *

class Player(BasePlayer):
    def __init__(self, timeLimit):
        BasePlayer.__init__(self, timeLimit)
        # Initialize table
        self._valueTable = {}
        for a in range(16):
            for b in range(16):
                for c in range(16):
                    for d in range(16):
                        self._valueTable[(a,b,c,d)] = random.uniform(0,1)
        
        self._learningRate = 0.05
        self._discountFactor = 0.90
        
        self._epsilon = 0.9
        self._epsilonMin = 0.1
        self._epsilonDecay = 0.999
    
        self._training = False
        self._totalGames = 0
        self._metrics = {
            'avgScores': [],
            'maxScores': [],
            'maxTiles': [],
            'avgUpdates': [],
            'epsilonHistory': []
        }
        self._current_period = {
            'scores': [],
            'max_score': 0,
            'max_tiles': [],
            'updates': []
        }

    def loadData(self, filename):
        try:
            with open(filename, 'rb') as dataFile:
                data = pickle.load(dataFile)
                
                # Support for loading both simple value tables and full data with tracking
                if isinstance(data, dict) and 'valueTable' in data:
                    self._valueTable = data['valueTable']
                    
                    # Load metrics if available
                    if 'metrics' in data:
                        self._metrics = data['metrics']
                        self._totalGames = data.get('totalGames', 0)
                        self._epsilon = data.get('epsilon', self._epsilon)
                else:
                    self._valueTable = data
            return True
        except (FileNotFoundError, Exception):
            return False

    def saveData(self, filename):
        data = {
            'valueTable': self._valueTable,
            'metrics': self._metrics,
            'totalGames': self._totalGames,
            'epsilon': self._epsilon
        }
        with open(filename, 'wb') as dataFile:
            pickle.dump(data, dataFile)

    def value(self, board):
        v = 0.
        for turns in range(4):
            g = board.rotate(turns)
            v += self._valueTable[tuple(g._board[:4])]
        return v

    def findMove(self, board):
        bestValue = float('-inf')
        bestMove = ''
        actions = board.actions()
        
        if not actions:
            self.setMove('')
            return
        
        # Epsilon-greedy
        if self._training and random.random() < self._epsilon:
            randomMove = random.choice(list(actions))
            self.setMove(randomMove)
            return
            
        # Otherwise, select the best
        for a in actions:
            v = 0
            for (g, p) in board.possibleResults(a):
                v += p * self.value(g)
            if v > bestValue:
                bestValue = v
                bestMove = a
        self.setMove(bestMove)

    def train(self, repetitions):
        """Train the agent for a specified number of games"""
        self._training = True
        
        for _ in range(repetitions):
            self._totalGames += 1
            
            # Decay epsilon
            if self._epsilon > self._epsilonMin:
                self._epsilon *= self._epsilonDecay
            
            state = Game2048()
            state.randomize()
            game_updates = []
            
            while not state.gameOver():
                self._startTime = time.time()
                self.findMove(state)
                move = self.getMove()
                if not move:
                    break
                    
                oldState = state
                state, reward = state.result(move)
                
                # Update the table
                update = self._learningRate * (reward + self._discountFactor*self.value(state) - self.value(oldState))
                game_updates.append(abs(update))
                
                for turns in range(4):
                    rotated = oldState.rotate(turns)
                    self._valueTable[tuple(rotated._board[:4])] += update
            
            # Track metrics
            score = state.getScore()
            max_tile = max(state._board) if state._board else 0
            
            self._current_period['scores'].append(score)
            self._current_period['max_score'] = max(self._current_period['max_score'], score)
            self._current_period['max_tiles'].append(max_tile)
            self._current_period['updates'].extend(game_updates)
            
            # Update metrics every 100 games
            if self._totalGames % 100 == 0:
                self._update_metrics()
        
        self._training = False
        return self._metrics
    
    def _update_metrics(self):
        """Update the tracking metrics with current period data"""
        if not self._current_period['scores']:
            return
            
        # Calculate period metrics
        avg_score = sum(self._current_period['scores']) / len(self._current_period['scores'])
        max_score = self._current_period['max_score']
        avg_max_tile = sum(self._current_period['max_tiles']) / len(self._current_period['max_tiles'])
        avg_update = sum(self._current_period['updates']) / len(self._current_period['updates']) if self._current_period['updates'] else 0
        
        # Add to metrics history
        self._metrics['avgScores'].append(avg_score)
        self._metrics['maxScores'].append(max_score)
        self._metrics['maxTiles'].append(avg_max_tile)
        self._metrics['avgUpdates'].append(avg_update)
        self._metrics['epsilonHistory'].append(self._epsilon)
        
        # Reset period tracking
        self._current_period = {
            'scores': [],
            'max_score': 0,
            'max_tiles': [],
            'updates': []
        }

    def evaluate(self, num_games=100):
        # No Exploration during evaluation
        self._training = False
        
        results = {
            'scores': [],
            'max_tiles': [],
        }
        
        for _ in range(num_games):
            state = Game2048()
            state.randomize()
            
            while not state.gameOver():
                self._startTime = time.time()
                self.findMove(state)
                move = self.getMove()
                if not move:
                    break
                state, _ = state.result(move)
            
            score = state.getScore()
            max_tile = max(state._board) if state._board else 0
            
            results['scores'].append(score)
            results['max_tiles'].append(max_tile)
        
        return results
    
    def get_metrics(self):
        """Return all tracked metrics"""
        return {
            'metrics': self._metrics,
            'totalGames': self._totalGames,
            'epsilon': self._epsilon
        }
