# ベースイメージを指定
FROM python:3.10-slim

WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    curl wget libgl1 libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
RUN pip install --no-cache-dir --upgrade pip

# ホストのファイルをコンテナにコピー
COPY . .

# Poetryをインストール
RUN curl -sSL https://install.python-poetry.org | python3

# Poetryのパスを設定
ENV PATH="/root/.local/bin:$PATH"

# プロジェクト内に仮想環境を作る
RUN poetry config virtualenvs.in-project true \
    && poetry install

# ポートを公開
EXPOSE 7860