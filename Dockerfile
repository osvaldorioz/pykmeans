# Usar Python 3.12 como imagen base
FROM python:3.12-slim

# Instalar herramientas necesarias para compilar C++ y dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación y los archivos de construcción
COPY . .

# Compilar la biblioteca de C++ con Pybind11
#RUN python setup.py build_ext --inplace
RUN c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) app/kmeans.cpp -o kmeans_module$(python3.12-config --extension-suffix)

# Exponer el puerto para Uvicorn
EXPOSE 8000

# Comando para correr la aplicación Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
