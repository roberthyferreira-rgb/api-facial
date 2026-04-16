from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
import face_recognition
import requests
import io

app = FastAPI(title="API de Reconhecimento Facial + Investigador")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# ROTA 1: A comparação manual que já fizemos (Para o App)
# ---------------------------------------------------------
@app.post("/comparar_rostos/")
async def comparar_rostos(
    foto_oficial: UploadFile = File(...), 
    foto_suspeita: UploadFile = File(...)
):
    try:
        oficial_bytes = await foto_oficial.read()
        suspeita_bytes = await foto_suspeita.read()

        oficial_img = face_recognition.load_image_file(io.BytesIO(oficial_bytes))
        suspeita_img = face_recognition.load_image_file(io.BytesIO(suspeita_bytes))

        locais_oficial = face_recognition.face_locations(oficial_img, number_of_times_to_upsample=1)
        locais_suspeita = face_recognition.face_locations(suspeita_img, number_of_times_to_upsample=1)

        oficial_encodings = face_recognition.face_encodings(oficial_img, known_face_locations=locais_oficial, num_jitters=1)
        suspeita_encodings = face_recognition.face_encodings(suspeita_img, known_face_locations=locais_suspeita, num_jitters=1)

        if not oficial_encodings:
            return {"erro": "Nenhum rosto encontrado na foto OFICIAL."}
        if not suspeita_encodings:
            return {"alerta": "Nenhum rosto detectado na foto suspeita."}

        meu_encoding = oficial_encodings[0]
        rosto_encontrado = False
        
        for rosto_desconhecido in suspeita_encodings:
            resultado = face_recognition.compare_faces([meu_encoding], rosto_desconhecido, tolerance=0.55)
            if resultado[0]:
                rosto_encontrado = True
                break

        if rosto_encontrado:
            return {"status": "alerta", "mensagem": "⚠️ ALERTA: Esta foto contém o seu rosto!"}
        else:
            return {"status": "limpo", "mensagem": "✅ Tudo limpo: Não é você nesta imagem."}

    except Exception as e:
        return {"erro": f"Erro técnico: {str(e)}"}


# ---------------------------------------------------------
# ROTA 2: O Novo Modo Caçador Automático
# ---------------------------------------------------------
@app.post("/cacar_na_internet/")
async def cacar_na_internet(
    termo_busca: str = Form(...),
    foto_oficial: UploadFile = File(...)
):
    try:
        # 1. Lê e memoriza o rosto oficial
        oficial_bytes = await foto_oficial.read()
        oficial_img = face_recognition.load_image_file(io.BytesIO(oficial_bytes))
        locais_oficial = face_recognition.face_locations(oficial_img, number_of_times_to_upsample=1)
        oficial_encodings = face_recognition.face_encodings(oficial_img, known_face_locations=locais_oficial, num_jitters=1)

        if not oficial_encodings:
            return {"erro": "Nenhum rosto encontrado na foto OFICIAL."}

        meu_encoding = oficial_encodings[0]
        
        # 2. Faz a varredura no DuckDuckGo
        resultados_positivos = []
        fotos_analisadas = 0
        
        # Limitamos a 5 resultados para não estourar a memória da Render gratuita
        with DDGS() as ddgs:
            resultados_busca = list(ddgs.images(termo_busca, max_results=5))

        # Disfarce para sites não bloquearem o download do nosso servidor
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        # 3. Baixa e analisa cada foto encontrada
        for img_data in resultados_busca:
            url_imagem = img_data.get('image')
            if not url_imagem: 
                continue
            
            try:
                # Tenta baixar a imagem (com limite de 5 segundos de espera)
                resposta = requests.get(url_imagem, headers=headers, timeout=5)
                if resposta.status_code == 200:
                    fotos_analisadas += 1
                    
                    suspeita_img = face_recognition.load_image_file(io.BytesIO(resposta.content))
                    locais_suspeita = face_recognition.face_locations(suspeita_img, number_of_times_to_upsample=1)
                    suspeita_encodings = face_recognition.face_encodings(suspeita_img, known_face_locations=locais_suspeita)
                    
                    # Compara com a foto da internet
                    for rosto_desconhecido in suspeita_encodings:
                        resultado = face_recognition.compare_faces([meu_encoding], rosto_desconhecido, tolerance=0.55)
                        if resultado[0]:
                            resultados_positivos.append(url_imagem) # Salva o link da foto!
                            break
            except Exception:
                continue # Se a foto for bloqueada ou o link estiver quebrado, pula pra próxima
        
        # 4. Devolve o relatório final
        if len(resultados_positivos) > 0:
            return {
                "status": "alerta",
                "mensagem": f"🚨 ALERTA! Rosto encontrado em {len(resultados_positivos)} foto(s).",
                "fotos_analisadas": fotos_analisadas,
                "links_encontrados": resultados_positivos
            }
        else:
            return {
                "status": "limpo",
                "mensagem": "✅ Tudo limpo. O buscador não encontrou o seu rosto.",
                "fotos_analisadas": fotos_analisadas,
                "links_encontrados": []
            }

    except Exception as e:
        return {"erro": f"Erro na investigação: {str(e)}"}
