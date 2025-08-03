FROM ollama/ollama:latest

# Copy the model pulling script
COPY pull-model.sh /pull-model.sh

RUN chmod +x /pull-model.sh \
    && /pull-model.sh

EXPOSE 11434