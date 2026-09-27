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
    relatorio = [] # 📋 O nosso Raio-X
    
    # 🎭 Disfarce Avançado (com Referer)
    headers_disfarce = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://www.facebook.com/"
    }

    for url in dados.urls:
        if not url.startswith("http"):
            continue
            
        try:
            resp = requests.get(url, headers=headers_disfarce, timeout=10)
            
            if resp.status_code == 200:
                img_fb = face_recognition.load_image_file(io.BytesIO(resp.content))
                encodings_fb = face_recognition.face_encodings(img_fb)

                if not encodings_fb:
                    relatorio.append({"foto": "📷 Imagem lida", "status": "⚠️ Nenhum rosto visível (muito pequena ou s/ rosto)"})
                    continue

                match_encontrado = False
                for encoding_suspeito in encodings_fb:
                    # Tolerância ajustada para 0.60
                    matches = face_recognition.compare_faces(rostos_conhecidos, encoding_suspeito, tolerance=0.60)
                    if True in matches:
                        fotos_encontradas.append(url)
                        match_encontrado = True
                        break
                
                if match_encontrado:
                    relatorio.append({"foto": "📷 Imagem lida", "status": "🚨 BINGO! VOCÊ FOI ENCONTRADO!"})
                else:
                    relatorio.append({"foto": "📷 Imagem lida", "status": f"❌ Encontrou {len(encodings_fb)} rosto(s), mas nenhum é o seu."})
            else:
                relatorio.append({"foto": "⛔ Falha", "status": f"O Facebook bloqueou o download (Erro {resp.status_code})"})
        except Exception as e:
            relatorio.append({"foto": "⛔ Falha", "status": "Erro na rede ao tentar descarregar."})

    return {
        "status": "concluido",
        "voce_aparece_em": len(fotos_encontradas),
        "links_com_voce": fotos_encontradas,
        "relatorio": relatorio # Enviamos o relatório de volta para o Chrome
    }
