tmux new-session -s gpu$GPUNUM 'tmux source-file ./.tmux.conf; $SHELL'
