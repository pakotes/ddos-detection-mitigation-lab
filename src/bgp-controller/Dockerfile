FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema e ExaBGP
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Instalar ExaBGP
RUN pip install exabgp

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p /etc/exabgp /var/run/exabgp logs

# Tornar script executável
RUN chmod +x start.sh

# Comando padrão
CMD ["./start.sh"]
