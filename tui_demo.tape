# calyxos full demo — core reactive engine + MLX backend
Output tui_demo.gif

Set Shell "zsh"
Set FontSize 13
Set Width 1100
Set Height 900
Set Padding 16
Set Theme "Dark+"
Set TypingSpeed 40ms
Set PlaybackSpeed 0.85

# Set up miniforge env for mlx (hidden from recording)
Hide
Type "export PATH=/Users/krish/miniforge3/bin:$PATH"
Enter
Sleep 1s
Type "clear"
Enter
Sleep 500ms
Show

# Part 1: Core reactive graph demo
Type "calyxos demo"
Sleep 300ms
Enter
Sleep 9s

# TUI: layered DAG view
Sleep 500ms
Type "flow"
Enter
Sleep 2500ms

# TUI: mutate an input, see invalidation cascade
Type "set input_a 99.0"
Enter
Sleep 2s

# TUI: flow with dirty markers
Type "flow"
Enter
Sleep 2500ms

# TUI: recompute only affected nodes
Type "eval output"
Enter
Sleep 1500ms

# TUI: dependency tree
Type "tree output"
Enter
Sleep 2500ms

Type "quit"
Enter
Sleep 500ms

# Part 2: MLX tensor backend
Type "calyxos mlx-demo --no-inspect"
Sleep 300ms
Enter
Sleep 10s

Sleep 2s
