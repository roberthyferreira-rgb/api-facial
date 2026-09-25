from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import requests
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

# Criamos um modelo para receber a lista de links da extensão
class ListaFotos(BaseModel):
    urls: list[str]

# 🧠 O CÉREBRO: Carrega suas fotos na memória quando o servidor liga
rostos_conhecidos = []
pasta_fotos = "minhas_fotos"

if os.path.exists(pasta_fotos):
    print("Carregando rostos base...")
    for nome_arquivo in os.listdir(pasta_fotos):
        caminho = os.path.join(pasta_fotos, nome_arquivo)
        try:
            img = face_recognition.load_image_file(caminho)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                rostos_conhecidos.append(encodings[0])
                print(f"Rosto de {nome_arquivo} carregado e memorizado!")
        except Exception as e:
            print(f"Erro ao ler {nome_arquivo}: {e}")

@app.post("/analisar_facebook/")
async def analisar_facebook(dados: ListaFotos):
    if not rostos_conhecidos:
        return {"erro": "Nenhum rosto base foi encontrado na pasta 'minhas_fotos'."}

    fotos_encontradas = []

    for url in dados.urls:
        try:
            # 1. O Python baixa a foto do Facebook
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                img_fb = face_recognition.load_image_file(io.BytesIO(resp.content))
                encodings_fb = face_recognition.face_encodings(img_fb)

                # 2. Verifica todos os rostos que estão na foto do Facebook
                for encoding_suspeito in encodings_fb:
                    # 3. Compara com os seus rostos memorizados
                    # Tolerância de 0.55 (Ajuste conforme necessário)
                    matches = face_recognition.compare_faces(rostos_conhecidos, encoding_suspeito, tolerance=0.55)
                    
                    if True in matches:
                        fotos_encontradas.append(url)
                        break # Achou você! Pula para a próxima foto
        except:
            continue # Se a foto der erro, ignora e segue a vida

    return {
        "status": "concluido",
        "total_analisado": len(dados.urls),
        "voce_aparece_em": len(fotos_encontradas),
        "links_com_voce": fotos_encontradas
    }
