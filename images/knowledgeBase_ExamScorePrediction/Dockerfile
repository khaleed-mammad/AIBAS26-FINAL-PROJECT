FROM busybox

# 1. Create a staging directory to store files safely
RUN mkdir -p /staging/knowledgeBase

# 2. Copy files to the STAGING directory (not /tmp directly yet)
COPY currentAiSolution.keras /staging/knowledgeBase/
COPY currentOlsSolution.pkl /staging/knowledgeBase/
COPY README.md /staging/knowledgeBase/

# 3. Create the volume directory just in case
RUN mkdir -p /tmp/knowledgeBase

# 4. COMMAND: When container starts, copy from staging to volume, then sleep
# We use 'cp -n' (no clobber) or just 'cp -r' to ensure data moves to the mounted volume
CMD cp -r /staging/* /tmp/ && echo "KnowledgeBase files copied to volume" && tail -f /dev/null