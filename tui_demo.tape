# calyxos demo — benchmark + TUI inspector
Output tui_demo.gif

Set Shell "zsh"
Set FontSize 13
Set Width 1100
Set Height 900
Set Padding 16
Set Theme "Dark+"
Set TypingSpeed 40ms
Set PlaybackSpeed 0.85

# Run the CLI
Type "calyxos demo"
Sleep 300ms
Enter
Sleep 9s

# TUI: show the layered graph
Sleep 500ms
Type "flow"
Enter
Sleep 2500ms

# TUI: inspect a node
Type "node output"
Enter
Sleep 2500ms

# TUI: set a value and see invalidation
Type "set input_a 99.0"
Enter
Sleep 2s

# TUI: flow again — see dirty markers
Type "flow"
Enter
Sleep 2500ms

# TUI: evaluate to trigger recomputation
Type "eval output"
Enter
Sleep 1500ms

# TUI: tree from output
Type "tree output"
Enter
Sleep 2500ms

Type "quit"
Enter
Sleep 500ms
