FROM busybox
RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

# Copy to specific subfolders as requested by subgoal 3
COPY training_data.csv /tmp/learningBase/train/
COPY test_data.csv /tmp/learningBase/validation/

# Copy everything else to the root of learningBase
COPY README.md /tmp/learningBase/


CMD tail -f /dev/null