# Usa um Linux levinho com Python já instalado
FROM python:3.10-slim

# Define a pasta de trabalho lá na nuvem
WORKDIR /app

# Instala as ferramentas que o motor dlib precisa para compilar (o que faltava no seu Windows)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as bibliotecas do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o seu código main.py para a nuvem
COPY . .

# Libera a porta da internet e liga o servidor
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]