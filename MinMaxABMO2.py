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

        if actions:
            self.setMove(actions[0])  # Default fallback
        else:
            self.setMove('U')
            return

        depth = 1
        bestMove = actions[0]
        bestScore = -10000

        while self.timeRemaining():
            self._depthCount += 1
            alpha = -float('inf')
            beta = float('inf')
            depthMove = None
            depthScore = -10000

            for a in actions:
                result = state.move(a)

                # SAFER no-op check
                if result._board == state._board:
                    continue

                if not self.timeRemaining():
                    break

                v = self.minPlayer(result, depth - 1, alpha, beta)
                if v is None:
                    break

                if v > depthScore:
                    depthScore = v
                    depthMove = a

                alpha = max(alpha, v)

            # Accept best move if we completed even part of the depth
            if depthMove is not None:
                bestMove = depthMove
                bestScore = depthScore
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

    def moveOrder(self, state):
        actions = state.actions()
        tile_weights = [
            4.0, 3.0, 2.0, 1.0,
            3.0, 1.0, 1.5, 1.0,
            2.0, 1.5, 1.0, 0.5,
            1.0, 0.5, 0.25, 0.1
        ]

        def gradient_score(board):
            return sum(board[i] * tile_weights[i] for i in range(16))

        scored = []
        for a in actions:
            next_state = state.move(a)
            scored.append((a, gradient_score(next_state._board)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scored]

    def heuristic(self, state):
        """Score-based heuristic using percent-scaled bonuses for features."""

        board = state._board
        base_score = state.getScore()

        # if base_score < 3000:
        #     gradient_pct = 0.1  # Stronger in early game
        # elif base_score < 10000:
        #     gradient_pct = 0.08
        # else:
        #     gradient_pct = 0.07

        # === Feature coefficients slightly more conservative agent ===
        SCORE_WEIGHT = 1.0
        CORNER_BONUS_PCT = 0.55  # 7% of score at most
        GRADIENT_BONUS_PCT = 0.12
        EMPTY_TILE_PCT = {
            'low': 0.02,   # full board
            'mid': 0.018,
            'high': 0.015
        }

        # === Compute base score (80% of raw score) ===
        value = base_score * SCORE_WEIGHT

        # === Corner anchoring bonus (scales with score) ===
        corner_score = self.improved_corner_score(state)  # Range: 0.0 to 1.0 (normalized)
        value += base_score * CORNER_BONUS_PCT * corner_score

        # === Empty tile contribution ===
        empty_tiles = sum(1 for cell in board if cell == 0)
        filled = 16 - empty_tiles
        if filled >= 12:
            tile_pct = EMPTY_TILE_PCT['low']
        elif filled >= 8:
            tile_pct = EMPTY_TILE_PCT['mid']
        else:
            tile_pct = EMPTY_TILE_PCT['high']

        empty_bonus = base_score * tile_pct * empty_tiles
        value += empty_bonus

        gradient_score = self.tile_gradient_score(state)
        value += base_score * GRADIENT_BONUS_PCT * gradient_score

        if base_score < 1000 and board[0] == max(board):
            value += 0.03 * base_score
        elif base_score < 2000 and board[0] == max(board):
            value += 0.02 * base_score


        return value

    def anchor_penalty(self, state):
        """Returns 1.0 if max tile is in top-left, else 0.0."""
        board = state._board
        return 1.0 if board[0] == max(board) else 0.0

    def monotonic_chain_score(self, state):
        """Normalized chain score for how long values decrease along top row and left column."""
        board = state._board

        # Top row: indices 0-1-2-3
        row_chain = 1
        for i in range(3):
            if board[i] >= board[i+1] and board[i+1] > 0:
                row_chain += 1
            else:
                break

        # Left column: indices 0,4,8,12
        col_chain = 1
        for i in range(3):
            top = board[4*i]
            below = board[4*(i+1)]
            if top >= below and below > 0:
                col_chain += 1
            else:
                break

        # Normalize: max value is 4
        return max(row_chain, col_chain) / 4.0

    def improved_corner_score(self, state):
        """Weighted score that rewards anchoring and maintaining monotonic chains from the top-left."""
        anchor = self.anchor_penalty(state)
        chain = self.monotonic_chain_score(state)
        return 0.8 * anchor + 0.2 * chain  # Tunable mix

    def tile_gradient_score(self, state):
        """Returns a weighted score based on tile values favoring top-left corner structure."""
        board = state._board
        tile_weights = [
            4.0, 3.0, 2.0, 1.0,
            3.0, 1.0, 1.5, 1.0,
            2.0, 1.5, 1.0, 0.5,
            1.0, 0.5, 0.25, 0.1
        ]

        weighted_sum = sum(board[i] * tile_weights[i] for i in range(16))

        # Normalize (optional): assume max tile 2048 near top left
        max_possible = 2048 * 4.0  # tile[0] is most valuable spot
        normalized = weighted_sum / max_possible

        return min(1.0, normalized)  # Keep within 0–1

    def stats(self):
        print(f'Average depth: {self._depthCount/self._count:.2f}')
        print(f'Branching factor: {self._childCount / self._parentCount:.2f}')
        print(f'Pruned nodes: {self._pruneCount}')
