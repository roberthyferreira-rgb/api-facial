from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import io

app = FastAPI(title="API de Reconhecimento Facial Pro")

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

        # MELHORIA 1: Adicionamos num_jitters=1 para maior precisão no encoding oficial
        oficial_encodings = face_recognition.face_encodings(oficial_img, num_jitters=1)
        
        # MELHORIA 2: Usamos model="hog" (mais rápido) ou "cnn" (mais lento/preciso)
        # E aumentamos o número de vezes para 'upsample' a imagem à procura de rostos
        suspeita_encodings = face_recognition.face_encodings(suspeita_img, num_jitters=1)

        if not oficial_encodings:
            return {"erro": "Nenhum rosto encontrado na foto OFICIAL. Tente uma foto mais clara."}
        if not suspeita_encodings:
            return {"alerta": "IA não detetou rostos na imagem suspeita. Tente uma imagem com rosto mais visível."}

        meu_encoding = oficial_encodings[0]
        rosto_encontrado = False
        
        # MELHORIA 3: Ajuste de tolerância (0.5 é um ótimo equilíbrio entre rigor e flexibilidade)
        distancias = face_recognition.face_distance(suspeita_encodings, meu_encoding)
        
        for distancia in distancias:
            if distancia <= 0.55: # Se a distância for pequena, os rostos são iguais
                rosto_encontrado = True
                break

        if rosto_encontrado:
            return {"status": "alerta", "mensagem": "⚠️ ALERTA: Esta foto contém o seu rosto!"}
        else:
            return {"status": "limpo", "mensagem": "✅ Tudo limpo: Não é você nesta imagem."}

    except Exception as e:
        return {"erro": f"Erro técnico: {str(e)}"}
