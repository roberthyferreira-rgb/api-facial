from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
import face_recognition
import requests
import io
import os

app = FastAPI(title="Caçador Facial Profissional")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛠️ CONFIGURAÇÃO DE SEGURANÇA
# No Render, você vai cadastrar essas duas variáveis nas configurações (Environment Variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

@app.post("/cacar_na_internet/")
async def cacar_na_internet(
    termo_busca: str = Form(...),
    foto_oficial: UploadFile = File(...)
):
    try:
        # 1. Processa o seu rosto oficial
        oficial_bytes = await foto_oficial.read()
        oficial_img = face_recognition.load_image_file(io.BytesIO(oficial_bytes))
        oficial_encodings = face_recognition.face_encodings(oficial_img, num_jitters=1)

        if not oficial_encodings:
            return {"erro": "Rosto não detectado na foto oficial."}
        
        meu_encoding = oficial_encodings[0]

        # 2. Busca no Google Imagens usando a API Oficial
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(
            q=termo_busca,
            cx=GOOGLE_CX,
            searchType="image",
            num=10  # Vamos buscar as 10 primeiras imagens
        ).execute()

        links_encontrados = []
        
        # 3. Analisa as imagens retornadas pelo Google
        if "items" in res:
            for item in res["items"]:
                url_img = item["link"]
                try:
                    # Baixa a imagem da internet
                    resp = requests.get(url_img, timeout=5)
                    if resp.status_code == 200:
                        img_internet = face_recognition.load_image_file(io.BytesIO(resp.content))
                        encodings_internet = face_recognition.face_encodings(img_internet)

                        # Compara a biometria
                        for encoding_suspeito in encodings_internet:
                            # Tolerância de 0.55 (Equilíbrio entre rigor e detecção)
                            match = face_recognition.compare_faces([meu_encoding], encoding_suspeito, tolerance=0.55)
                            if match[0]:
                                links_encontrados.append(url_img)
                                break
                except:
                    continue

        # 4. Resultado Final
        if links_encontrados:
            return {
                "status": "alerta",
                "mensagem": f"🚨 Encontrado em {len(links_encontrados)} sites!",
                "links": links_encontrados
            }
        
        return {"status": "limpo", "mensagem": "✅ Nenhuma correspondência oficial encontrada."}

    except Exception as e:
        return {"erro": f"Erro na busca Google: {str(e)}"}
