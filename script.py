import os
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing import image


# =====================================
# CONFIGURAÇÕES
# =====================================

PASTA_IMAGENS = "imagens"
IMAGEM_CONSULTA = "consulta.jpg"
TOP_N = 5

IMG_SIZE = (224, 224)


# =====================================
# CARREGA MODELO PRÉ-TREINADO
# =====================================

print("Carregando modelo...")

modelo = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


# =====================================
# EXTRAIR FEATURES
# =====================================

def extrair_feature(caminho_imagem):

    img = image.load_img(
        caminho_imagem,
        target_size=IMG_SIZE
    )

    x = image.img_to_array(img)

    x = np.expand_dims(
        x,
        axis=0
    )

    x = preprocess_input(x)

    feature = modelo.predict(
        x,
        verbose=0
    )[0]

    return feature


# =====================================
# PROCESSAR BASE DE IMAGENS
# =====================================

print("Processando imagens da base...")

embeddings = []
arquivos = []

extensoes_validas = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

for arquivo in os.listdir(PASTA_IMAGENS):

    if not arquivo.lower().endswith(extensoes_validas):
        continue

    caminho = os.path.join(
        PASTA_IMAGENS,
        arquivo
    )

    try:

        feature = extrair_feature(caminho)

        embeddings.append(feature)
        arquivos.append(arquivo)

        print(f"OK -> {arquivo}")

    except Exception as e:

        print(
            f"Erro ao processar {arquivo}: {e}"
        )

if len(embeddings) == 0:
    raise Exception(
        "Nenhuma imagem encontrada na pasta."
    )

embeddings = np.array(embeddings)

print(
    f"\nTotal de imagens carregadas: {len(arquivos)}"
)


# =====================================
# CONSULTA
# =====================================

print("\nAnalisando imagem de consulta...")

feature_consulta = extrair_feature(
    IMAGEM_CONSULTA
)

similaridades = cosine_similarity(
    [feature_consulta],
    embeddings
)[0]

indices = np.argsort(
    similaridades
)[::-1]


# =====================================
# RESULTADO
# =====================================

print("\n" + "=" * 50)
print("PRODUTOS MAIS SEMELHANTES")
print("=" * 50)

for posicao, indice in enumerate(
    indices[:TOP_N],
    start=1
):

    print(
        f"{posicao}. "
        f"{arquivos[indice]} "
        f"(similaridade: "
        f"{similaridades[indice]:.4f})"
    )

print("\nPesquisa concluída.")
