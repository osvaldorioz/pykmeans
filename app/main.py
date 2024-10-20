from fastapi import FastAPI
import kmeans_module
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

@app.post("/kmeans")
async def km(num_clusters: int, 
             iteraciones: int, 
             data: Matrix):
    start = time.time()
    
    # Inicializar el modelo con 2 clusters y un máximo de 100 iteraciones
    kmeans = kmeans_module.KMeans(2, 200)

    # Ajustar el modelo a los datos
    kmeans.fit(data.matrix)

    # Predecir los clusters de los datos de entrada
    labels = kmeans.predict(data.matrix)
    #print("Labels:", labels)
    str1 = f'{labels}'

    # Obtener los centroides finales
    centroides = kmeans.get_centroids()
    #print("Centroids:", centroides)
    str2 = f'{centroides}'

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Labels": str1,
        "Centroides": str2
    }
    jj = json.dumps(j1)

    return jj