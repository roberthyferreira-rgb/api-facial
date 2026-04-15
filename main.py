from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import io

app = FastAPI(title="API de Reconhecimento Facial Turbo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

       # Trocando de 2 para 1 para salvar a memória RAM do servidor gratuito
        locais_oficial = face_recognition.face_locations(oficial_img, number_of_times_to_upsample=1)
        locais_suspeita = face_recognition.face_locations(suspeita_img, number_of_times_to_upsample=1)
        # Agora passamos os locais encontrados para extrair a biometria
        oficial_encodings = face_recognition.face_encodings(oficial_img, known_face_locations=locais_oficial, num_jitters=1)
        suspeita_encodings = face_recognition.face_encodings(suspeita_img, known_face_locations=locais_suspeita, num_jitters=1)

        if not oficial_encodings:
            return {"erro": "Nenhum rosto encontrado na foto OFICIAL."}
        if not suspeita_encodings:
            return {"alerta": "Ainda não consegui detetar rostos, mesmo com zoom máximo."}

        meu_encoding = oficial_encodings[0]
        rosto_encontrado = False
        
        # Compara com todos os rostos encontrados na foto suspeita
        for rosto_desconhecido in suspeita_encodings:
            # Tolerância de 0.55 (mais rigoroso para evitar falsos positivos)
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
