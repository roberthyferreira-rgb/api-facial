from fastapi import FastAPI, UploadFile, File
import face_recognition
import io

app = FastAPI(title="API de Reconhecimento Facial")

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

        oficial_encodings = face_recognition.face_encodings(oficial_img)
        suspeita_encodings = face_recognition.face_encodings(suspeita_img)

        if not oficial_encodings:
            return {"erro": "Nenhum rosto encontrado na sua foto OFICIAL."}
        if not suspeita_encodings:
            return {"alerta": "Nenhum rosto humano detectado na foto da internet."}

        meu_encoding = oficial_encodings[0]
        rosto_encontrado = False
        
        for rosto_desconhecido in suspeita_encodings:
            resultado = face_recognition.compare_faces([meu_encoding], rosto_desconhecido, tolerance=0.5)
            if resultado[0]:
                rosto_encontrado = True
                break

        if rosto_encontrado:
            return {"status": "alerta", "mensagem": "⚠️ ALERTA: Esta foto contém o seu rosto!"}
        else:
            return {"status": "limpo", "mensagem": "✅ Tudo limpo: Não é você nesta imagem."}

    except Exception as e:
        return {"erro": f"Falha ao processar as imagens: {str(e)}"}