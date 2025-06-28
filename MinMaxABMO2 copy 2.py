from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

		self._nodeCount = 0
		self._parentCount = 0
		self._childCount = 0
		self._depthCount = 0
		self._count = 0
		self._pruneCount = 0  # Track pruned branches

	def findMove(self, state):
		self._count += 1
		actions = self.moveOrder(state)
		depth = 1
		MAX_DEPTH = 10
		while self.timeRemaining() and depth <= MAX_DEPTH:
			self._depthCount += 1
			self._parentCount += 1
			self._nodeCount += 1
			best = -10000
			alpha = -float('inf')  # Initialize alpha for root
			beta = float('inf')    # Initialize beta for root
			bestMove = None
			for a in actions:
				result = state.move(a)
				if not self.timeRemaining(): return
				v = self.minPlayer(result, depth-1, alpha, beta)
				if v is None: return
				if v > best:
					best = v
					bestMove = a
				alpha = max(alpha, best)  # Update alpha at root
						
			self.setMove(bestMove)

			depth += 1

	def maxPlayer(self, state, depth, alpha, beta):
		# The max player gets to choose the move
		self._nodeCount += 1
		self._childCount += 1

		if state.gameOver():
			return state.getScore()
			
		if depth == 0:
			return self.heuristic(state)
			
		actions = self.moveOrder(state)
		self._parentCount += 1
		best = -10000
		
		for a in actions:
			if not self.timeRemaining(): return None
			result = state.move(a)
			v = self.minPlayer(result, depth-1, alpha, beta)
			if v is None: return None
			if v > best:
				best = v
				
			alpha = max(alpha, best)  # Update alpha
			if alpha >= beta:
				self._pruneCount += 1  # Count pruned branch
				break  # Beta cutoff
				
		return best

	def minPlayer(self, state, depth, alpha, beta):
		# The min player chooses where to add the extra tile and whether it is a 2 or a 4
		self._nodeCount += 1
		self._childCount += 1

		if state.gameOver():
			return state.getScore()
			
		if depth == 0:
			return self.heuristic(state)

		self._parentCount += 1
		best = 1e6
		
		for (t,v) in state.possibleTiles():
			if not self.timeRemaining(): return None
			result = state.addTile(t,v)
			v = self.maxPlayer(result, depth-1, alpha, beta)
			if v is None: return None
			if v < best:
				best = v
				
			beta = min(beta, best)  # Update beta
			if alpha >= beta:
				self._pruneCount += 1
				break  # Alpha cutoff

		return best

	def heuristic(self, state):
		"""Normalized, trainable evaluation function scaled to match real game scores."""

		board = state._board
		raw_score = state.getScore()
		empty_tiles = sum(1 for cell in board if cell == 0)
		corner_value = self.corner_anchoring(state)
		fullness = (16 - empty_tiles) / 16.0

		# Normalize features
		normalized = {
			'score': raw_score / 30000.0,  # Normalize to target max score
			'corners': corner_value / 20.0,  # Assume corner bonus maxes at 20
			'empty': empty_tiles / 16.0,
			'danger': fullness**2           # Nonlinear urgency when board is full
		}

		# Tuned weights for untrained but effective behavior
		weights = {
			'score': 18.0,
			'corners': 5.0,
			'empty': 8.0,
			'danger': -4.0  # Penalize full boards (opposite of empty)
		}

		# Weighted sum of normalized features
		value = sum(normalized[key] * weights[key] for key in weights)

		# Scale back into real-world score territory
		# Target peak around ~25–30k, typical range ~10–20k
		return value * 2000
        
	def moveOrder(self, state):
	
		actions = state.actions()
		scored_actions = []
		
		# Scale empty tile value based on the largest tile on the board
		max_tile_value = max(state._board)
		empty_tile_value = max(16, max_tile_value / 8)
		
		for a in actions:
			next_state = state.move(a)
			# score_gain = next_state.getScore() - state.getScore()
			score = next_state.getScore()
			empty_tiles = sum(1 for cell in next_state._board if cell == 0)
			
			# Dynamic weighting based on game progression
			total_score = score + (empty_tiles * empty_tile_value)
			scored_actions.append((a, total_score))
		
		scored_actions.sort(key=lambda x: x[1], reverse=True)
		return [a for a, _ in scored_actions]
    
	def corner_anchoring(self, state):
		# Prioritize moves that keep the highest tiles in corners
		board = state._board
		# Find max tile
		max_val = max(board)
		max_idx = board.index(max_val)

		# Check if max is in corner
		corners = [0, 3, 12, 15]
		if max_idx not in corners:
			return 0

		score = 10  # Bonus for corner max tile

		# Identify adjacent indices
		if max_idx == 0:  # Top-left
			adjacent = [1, 4]
		elif max_idx == 3:  # Top-right
			adjacent = [2, 7]
		elif max_idx == 12:  # Bottom-left
			adjacent = [8, 13]
		else:  # Bottom-right (15)
			adjacent = [11, 14]

		# Score decreasing pattern from corner
		for adj in adjacent:
			if board[adj] > 0 and board[adj] <= board[max_idx]:
				score += 5

		return score

	def stats(self):
		print(f'Average depth: {self._depthCount/self._count:.2f}')
		print(f'Branching factor: {self._childCount / self._parentCount:.2f}')
		print(f'Pruned nodes: {self._pruneCount}')
