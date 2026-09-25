import face_recognition
import os
import pickle

pasta_fotos = "minhas_fotos"
rostos_conhecidos = []

print("🧠 Iniciando o treinamento da sua fisionomia...")

# Garante que a pasta existe antes de começar
if not os.path.exists(pasta_fotos):
    print(f"❌ Erro: Crie uma pasta chamada '{pasta_fotos}' aqui e coloque suas fotos dentro.")
    exit()

# Lê cada foto e extrai os bytes matemáticos
for nome_arquivo in os.listdir(pasta_fotos):
    if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        caminho = os.path.join(pasta_fotos, nome_arquivo)
        try:
            img = face_recognition.load_image_file(caminho)
            encodings = face_recognition.face_encodings(img)
            
            if encodings:
                rostos_conhecidos.append(encodings[0])
                print(f"✅ Rosto de {nome_arquivo} mapeado com sucesso!")
            else:
                print(f"⚠️ Nenhum rosto encontrado na foto {nome_arquivo}.")
        except Exception as e:
            print(f"❌ Erro ao ler a foto {nome_arquivo}: {e}")

# Salva a matriz matemática num arquivo de dados compacto
with open("mapa_facial.dat", "wb") as arquivo_dados:
    pickle.dump(rostos_conhecidos, arquivo_dados)

print("\n🎉 Treinamento concluído! O arquivo 'mapa_facial.dat' foi gerado.")
print("Agora você só precisa enviar esse arquivo .dat para o GitHub!")
