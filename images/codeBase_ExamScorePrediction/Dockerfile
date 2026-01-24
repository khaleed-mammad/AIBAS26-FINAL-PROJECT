FROM busybox
RUN mkdir -p /tmp/codeBase/
COPY activation_data.csv /tmp/codeBase/
COPY apply_ann.py /tmp/codeBase/
COPY apply_ols.py /tmp/codeBase/
COPY README.md /tmp/codeBase/
CMD ["sh"]