from kaggle_environments import make
from htmrl_agent import agent_fn

def play_one_game():
    print("Initializing environment...")
    # Initialize orbit_wars environment
    env = make("orbit_wars", debug=True)
    
    print("Running a single game...")
    env.run([agent_fn, agent_fn])
    
    print("\nGame Over!")
    
    # Print the final result
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
        
    # Render and view the game replay immediately in the browser
    try:
        import tempfile
        import webbrowser
        import os
        
        out_html = env.render(mode="html")
        
        # Create a temporary file that is automatically deleted when closed
        fd, temp_path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(out_html)
            
        print("\nOpening replay in your default web browser...")
        webbrowser.open('file://' + temp_path)
    except Exception as e:
        print(f"\nCould not generate HTML replay: {e}")

if __name__ == "__main__":
    play_one_game()
