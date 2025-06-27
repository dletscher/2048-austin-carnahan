import argparse
import importlib.util
import time
from Game2048 import Game2048

def load_agent(agent_file, time_limit):
    """Load agent module and instantiate the player"""
    module_name = agent_file.replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Player(time_limit)

def run_game(agent, silent=True):
    """Run a single game and return results"""
    state = Game2048()
    state.randomize()
    
    moves = 0
    while not state.gameOver():
        if not silent:
            print(state)
        
        agent._startTime = time.time()
        agent.findMove(state)
        move = agent.getMove()
        
        if not silent:
            print(f'Player moves {move}\n')
            
        state, reward = state.result(move)
        moves += 1
    
    final_score = state.getScore()
    max_tile = max(state._board)
    
    if not silent:
        print(f"Game over! Score: {final_score}, Max Tile: 2^{max_tile}")
        
    return {
        'score': final_score,
        'max_tile': max_tile,
        'moves': moves
    }

def evaluate_agent(agent_file, games, time_limit, verbose=False):
    """Run multiple games and collect statistics"""
    print(f"Evaluating {agent_file} for {games} games (time limit: {time_limit}s per move)")
    
    agent = load_agent(agent_file, time_limit)
    
    # Initialize statistics
    scores = []
    max_tiles = []
    total_moves = 0
    start_time = time.time()
    
    # Run games
    for i in range(games):
        game_start = time.time()
        
        # Print progress indicator
        if verbose:
            print(f"\nGame {i+1}/{games}")
        else:
            print(f"Running game {i+1}/{games}...", end='\r')
            
        # Run the game
        result = run_game(agent, silent=not verbose)
        
        # Record statistics
        scores.append(result['score'])
        max_tiles.append(result['max_tile'])
        total_moves += result['moves']
        
        game_time = time.time() - game_start
        
        if verbose:
            print(f"Game {i+1} completed: Score={result['score']}, "
                  f"Max Tile=2^{result['max_tile']} ({2**result['max_tile']}), "
                  f"Time={game_time:.1f}s")
    
    # Calculate final statistics
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    avg_moves = total_moves / games
    
    # Get branching factor and depth stats from agent
    agent.stats()  # This will print the stats directly
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Games played: {games}")
    print(f"Average score: {avg_score:.1f}")
    print(f"Top score: {max_score}")
    print(f"Average moves per game: {avg_moves:.1f}")
    print(f"Total evaluation time: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    # Show max tile distribution
    print("\nMax tile distribution:")
    tile_counts = {}
    for tile in max_tiles:
        tile_val = 2**tile
        tile_counts[tile_val] = tile_counts.get(tile_val, 0) + 1
        
    for tile, count in sorted(tile_counts.items()):
        percentage = (count / games) * 100
        print(f"  {tile}: {count} games ({percentage:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2048 agent performance")
    parser.add_argument("agent", type=str, help="Path to agent file")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--time", type=float, default=1.0, help="Time limit per move in seconds")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    evaluate_agent(args.agent, args.games, args.time, args.verbose)