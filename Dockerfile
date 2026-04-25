FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install torch CPU-only first (smaller and faster)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV AGENT_MODE=heuristic
ENV MODEL_PATH=macos-llm-clean

EXPOSE 7860

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]

