# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13
FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}-slim AS builder

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_HOME='/opt/poetry'
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=true

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean \
      && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
        > /etc/apt/apt.conf.d/keep-cache

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        g++

RUN \
      --mount=type=cache,target=/root/.cache \
      /usr/local/bin/python -m pip install --upgrade pip poetry

RUN \
      --mount=type=cache,target=/root/.cache \
      --mount=type=bind,source=.,target=/mnt/host \
      cp -a /mnt/host /tmp/gcmg \
      && /usr/local/bin/python -m poetry --directory=/tmp/gcmg build --format=wheel \
      && /usr/local/bin/python -m pip install --upgrade /tmp/gcmg/dist/gcmg-*.whl


FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}-slim AS cli

ARG USER_NAME=gcmg
ARG USER_UID=1001
ARG USER_GID=1001

COPY --from=builder /usr/local /usr/local
COPY --from=builder /etc/apt/apt.conf.d/keep-cache /etc/apt/apt.conf.d/keep-cache

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN \
      rm -f /etc/apt/apt.conf.d/docker-clean

RUN \
      --mount=type=cache,target=/var/cache/apt,sharing=locked \
      --mount=type=cache,target=/var/lib/apt,sharing=locked \
      apt-get -y update \
      && apt-get -y upgrade \
      && apt-get -y install --no-install-recommends --no-install-suggests \
        ca-certificates git

RUN \
      groupadd --gid "${USER_GID}" "${USER_NAME}" \
      && useradd --uid "${USER_UID}" --gid "${USER_GID}" --shell /bin/bash --create-home "${USER_NAME}"

USER "${USER_NAME}"

HEALTHCHECK NONE

ENTRYPOINT ["/usr/local/bin/gcmg"]
