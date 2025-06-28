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
		while self.timeRemaining():
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
		"""Emphasizes maintaining empty tiles while still considering structure"""
		# Base score (reduced weight)
		score = state.getScore() * 0.8

		# Corner anchoring - slightly reduced
		corner_value = self.corner_anchoring(state) * 20

		# Empty tiles - significantly increased weight
		empty_tiles = sum(1 for cell in state._board if cell == 0)

		# Dynamic empty tile weighting based on board density
		# When board is getting full, empty tiles become extremely valuable
		filled = 16 - empty_tiles
		if filled >= 12:  # 75%+ full board
			empty_weight = 300  # Critical to maintain space
		elif filled >= 8:  # 50%+ full board
			empty_weight = 250  # Very important
		else:  # More open board
			empty_weight = 200  # Still important

		empty_value = empty_tiles * empty_weight

		# Combined evaluation with empty-focused weights
		return score + corner_value + empty_value

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
		# Prioritize moves that keep the highest tiles in corners with directional preference
		board = state._board

		# Find max tile
		max_val = max(board)
		max_idx = board.index(max_val)

		# Base score
		score = 0

		# Check if max is in corner
		corners = [0, 3, 12, 15]
		if max_idx not in corners:
			return 0

		# Corner preference - top-left (0) is most preferred
		corner_weights = {0: 15, 3: 10, 12: 10, 15: 5}  # Weight by strategic preference
		score += corner_weights[max_idx]  # Higher bonus for preferred corners

		# Identify adjacent indices
		if max_idx == 0:  # Top-left (preferred corner)
			adjacent = [1, 4]
			diag = 5  # Diagonal adjacent

			# Check full row and column for decreasing pattern from top-left
			row_pattern = 0
			for c in range(3):
				if board[c] >= board[c+1] and board[c] > 0:
					row_pattern += 2

			col_pattern = 0
			for r in range(3):
				if board[4*r] >= board[4*(r+1)] and board[4*r] > 0:
					col_pattern += 2
			score += row_pattern + col_pattern

		elif max_idx == 3:  # Top-right
			adjacent = [2, 7]
			diag = 6
		elif max_idx == 12:  # Bottom-left
			adjacent = [8, 13]
			diag = 9
		else:  # Bottom-right (15)
			adjacent = [11, 14]
			diag = 10

		# Score decreasing pattern from corner
		for adj in adjacent:
			if board[adj] > 0 and board[adj] <= board[max_idx]:
				score += 5

		# Check diagonal adjacent
		if board[diag] > 0 and board[diag] <= min(board[adjacent[0]], board[adjacent[1]]):
			score += 3  # Bonus for smooth diagonal transition

		return score

	def stats(self):
		print(f'Average depth: {self._depthCount/self._count:.2f}')
		print(f'Branching factor: {self._childCount / self._parentCount:.2f}')
		print(f'Pruned nodes: {self._pruneCount}')
