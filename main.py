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

# 🧠 O CÉREBRO OTIMIZADO: Carrega os bytes pré-calculados em milissegundos
rostos_conhecidos = []
arquivo_mapa = "mapa_facial.dat"

if os.path.exists(arquivo_mapa):
    print("Carregando mapa facial pré-calculado...")
    with open(arquivo_mapa, "rb") as arquivo_dados:
        rostos_conhecidos = pickle.load(arquivo_dados)
    print(f"✅ Cérebro ativado com {len(rostos_conhecidos)} faces memorizadas!")
else:
    print("⚠️ Arquivo mapa_facial.dat não encontrado.")

# ... (Mantenha o resto da sua rota @app.post("/analisar_facebook/") exatamente igual) ...
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
