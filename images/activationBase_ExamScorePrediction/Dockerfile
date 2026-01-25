FROM busybox

# 1. Create staging
RUN mkdir -p /staging/activationBase

# 2. Copy to staging
COPY activation_data.csv /staging/activationBase/
COPY README.md /staging/activationBase/

# 3. Create volume directory
RUN mkdir -p /tmp/activationBase

# 4. COMMAND: Copy runtime -> wait
CMD cp -r /staging/* /tmp/ && echo "ActivationBase files copied to volume" && tail -f /dev/null