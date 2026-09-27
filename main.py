from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import requests
import pickle
import io
import os

app = FastAPI(title="Cérebro Detetive do Facebook")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ListaFotos(BaseModel):
    urls: list[str]

rostos_conhecidos = []
arquivo_mapa = "mapa_facial.dat"

if os.path.exists(arquivo_mapa):
    print("Carregando mapa facial pré-calculado...")
    with open(arquivo_mapa, "rb") as arquivo_dados:
        rostos_conhecidos = pickle.load(arquivo_dados)

@app.post("/analisar_facebook/")
async def analisar_facebook(dados: ListaFotos):
    if not rostos_conhecidos:
        return {"erro": "O mapa facial não foi carregado."}

    fotos_encontradas = []
    
    # 🎭 O DISFARCE: Engana o Facebook fingindo ser um navegador real
    headers_disfarce = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
    }

    for url in dados.urls:
        if not url.startswith("http"):
            continue # Ignora links defeituosos
            
        try:
            # Baixa a foto usando o disfarce
            resp = requests.get(url, headers=headers_disfarce, timeout=10)
            
            if resp.status_code == 200:
                img_fb = face_recognition.load_image_file(io.BytesIO(resp.content))
                encodings_fb = face_recognition.face_encodings(img_fb)

                for encoding_suspeito in encodings_fb:
                    # 🔧 Ajuste de Tolerância para 0.60 (Mais flexível para fotos do Facebook)
                    matches = face_recognition.compare_faces(rostos_conhecidos, encoding_suspeito, tolerance=0.60)
                    
                    if True in matches:
                        fotos_encontradas.append(url)
                        break
        except:
            continue # Se a foto der erro de download, segue a vida

    return {
        "status": "concluido",
        "total_analisado": len(dados.urls),
        "voce_aparece_em": len(fotos_encontradas),
        "links_com_voce": fotos_encontradas
    }
