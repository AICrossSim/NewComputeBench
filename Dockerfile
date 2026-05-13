FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Cache dependency layer: only re-resolved when lock or project metadata change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy full source (submodules must be checked out before `docker build`)
COPY . .

# Install the project itself and the mase editable submodule
RUN uv sync --frozen && \
    uv pip install --no-deps -e ./submodules/mase

ENTRYPOINT ["uv", "run"]
CMD ["python"]
