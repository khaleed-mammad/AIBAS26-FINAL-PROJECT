FROM python:3.10-slim
RUN pip install tensorflow pandas numpy scikit-learn statsmodels
WORKDIR /app
# Explicitly copy the files
COPY apply_ann.py .
COPY apply_ols.py .
COPY README.md .
# Set log flushing
ENV PYTHONUNBUFFERED=1
CMD ["python3", "apply_ann.py"]