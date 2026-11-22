# Dockerfile  [Serving image: the API with the released model]
# -------------------------------------------------------------------
# Build     docker build -t api .   (models/model.joblib must exist: dvc repro, or the prod model in deploy)
# Run       docker run -p 8000:8000 api, then open http://localhost:8000/docs
# Packages  exact versions from conda-lock.yml (linux-64), the same lock CI tests with
# Context   only what .dockerignore allows: src/, config.yaml, conda-lock.yml and the model
# Platform  always linux/amd64, the platform of the lock, CI and Render (Apple chips run it through emulation)

FROM --platform=linux/amd64 mambaorg/micromamba:2.3.3

# 1. Locked environment first: this slow layer is cached until conda-lock.yml changes
COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /tmp/conda-lock.yml
RUN micromamba install --yes --name base --file /tmp/conda-lock.yml \
    && micromamba clean --all --yes \
    && rm /tmp/conda-lock.yml

# 2. App folder owned by the non-root user, so the API can write logs/pipeline.log
USER root
RUN mkdir /app && chown $MAMBA_USER:$MAMBA_USER /app
USER $MAMBA_USER
WORKDIR /app

# 3. Code, settings and the released model; nothing else from the repo
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ src/
COPY --chown=$MAMBA_USER:$MAMBA_USER config.yaml config.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER models/model.joblib models/model.joblib

# 4. Render sets PORT; locally it defaults to 8000
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
