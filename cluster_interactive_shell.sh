#!/bin/sh

# --partition 'adlr_interactive_32GB'
# --partition 'batch_32GB'


submit_job --cpu 10 --gpu 1 \
       --partition 'adlr_interactive_32GB' \
       --mounts "$MOUNTS" \
       --workdir "$SRC_DIR/fots" \
       --image `cat docker_image` \
       --coolname \
       --interactive \
       -c "bash"
