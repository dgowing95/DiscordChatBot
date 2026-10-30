FROM python:3.14-slim

# Browsers go to a fixed path, not root's home, so the non-root user below
# can run them.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY requirements.txt .
# --only-shell: the headless shell is all a renderer needs, and much smaller
# than full Chromium. --with-deps installs the system libraries it links.
RUN pip --no-cache-dir install -r requirements.txt \
    && playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 browser
WORKDIR /app
COPY . /app
USER 10001

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
