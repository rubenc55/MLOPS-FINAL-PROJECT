# Dockerfile  [Serving image: the API with the released model]
# -------------------------------------------------------------------
# Build     docker build -t api .   (models/model.joblib must exist: dvc repro, or the prod model in deploy)
#           deploy.yml adds --build-arg APP_VERSION=<release tag> GIT_COMMIT=<commit>, shown by /health
# Run       docker run -p 8000:8000 api, then open http://localhost:8000/docs
# Packages  exact versions from conda-lock.yml (linux-64), the same lock CI tests with
# Context   only what .dockerignore allows: src/, config.yaml, conda-lock.yml, the model and its lineage
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

# 3. Code, settings and the released model (with model_info.json when deploy fetched the prod model)
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ src/
COPY --chown=$MAMBA_USER:$MAMBA_USER config.yaml config.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER models/ models/

# 4. Release tag and code commit for /health; Render sets PORT, locally it defaults to 8000
ARG APP_VERSION=dev
ARG GIT_COMMIT=unknown
ENV APP_VERSION=$APP_VERSION GIT_COMMIT=$GIT_COMMIT
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
