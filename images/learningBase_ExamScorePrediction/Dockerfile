FROM busybox

RUN mkdir -p /tmp/learningBase/train/ /tmp/learningBase/validation/

COPY training_data.csv /tmp/learningBase/train/
COPY test_data.csv /tmp/learningBase/validation/
COPY README.md /tmp/learningBase/

# Keep the container from exiting immediately if run
CMD ["sh"]