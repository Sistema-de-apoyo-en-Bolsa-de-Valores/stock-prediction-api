# stock-prediction-api (Puerto 8081)

API de predicción del precio de las acciones desarrollada en Python y FastAPI.

## Pasos para levantar en local

### 1. Crear archivo .env en la raíz del proyecto

![image](https://github.com/user-attachments/assets/271a4380-37bd-44a2-b5a3-c43539acef9b)

```bash
TF_ENABLE_ONEDNN_OPTS=0
```

### 2. Instalar librerías

```bash
pip install -r requirements.txt
```

### 3. Levantar el proyecto en el puerto 8081

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8081
```
