# A única responsabilidade deste arquivo é realizar a tarefa de segmentação.
# Segmentar, neste caso, significa criar uma "máscara" ou "molde" 3D que 
# destaca a região exata que queremos analisar (a nossa Região de Interesse, ou ROI).

# --- Importações de Bibliotecas ---
# Importamos as ferramentas necessárias para a cirurgia.

# SimpleITK é a nossa principal biblioteca para manipulação de imagens médicas.
# Ela entende formatos como DICOM e NIfTI e sabe como lidar com as informações
# espaciais (tamanho do pixel, orientação da imagem, etc.).
import SimpleITK as sitk

# NumPy é a biblioteca fundamental para computação numérica em Python.
# Nós a usamos para fazer os cálculos matemáticos de forma rápida e eficiente
# sobre as matrizes de pixels que compõem a imagem.
import numpy as np


# Definimos uma classe 'Segmentation' para organizar nossa função.
# Isso torna o código mais limpo e profissional.
class Segmentation():
    
    # O '@staticmethod' significa que esta função pertence à classe, mas não precisa
    # de uma 'instância' para ser chamada. Podemos simplesmente chamar 
    # 'Segmentation.create_image_mask(...)' de qualquer lugar do nosso projeto.
    @staticmethod
    def create_image_mask(image_sitk: sitk.Image, center_mm: list, radius_mm: int = 5) -> sitk.Image:
        """
        cria uma máscara binária esferica 3D em memoria a partir de uma imagem de referencia
        """

        # --- Passo 1: Converter Coordenadas do Mundo Real para Coordenadas de Voxel ---
        # As coordenadas da planilha (-25.7, 31.8, -38.5) estão em milímetros (espaço físico).
        # Para "pintar" a esfera na imagem, precisamos saber em qual pixel (voxel) esse ponto cai.
        # Esta função do SimpleITK faz exatamente essa conversão para nós.
        voxel_center_continuous = image_sitk.TransformPhysicalPointToContinuousIndex(center_mm)

        # --- Passo 2: Preparar os Dados para o Cálculo da Distância ---
        # Cria três matrizes 3D (zz, yy, xx) que contêm, para cada voxel, sua própria coordenada
        # de índice. Ex: o voxel no canto (0,0,0) terá os valores 0,0,0 nessas matrizes.
        zz, yy, xx = np.mgrid[:image_sitk.GetDepth(), :image_sitk.GetHeight(), :image_sitk.GetWidth()]
        
        # Pega o "espaçamento" (spacing) da imagem original.
        # O espaçamento nos diz o tamanho de cada voxel em milímetros (ex: 0.5mm, 0.5mm, 3.0mm).
        # Os voxels raramente são cubos perfeitos, então precisamos corrigir a distância em cada eixo.
        spacing = np.array(image_sitk.GetSpacing())
        
        # --- Passo 3: Calcular a Distância e Criar a Esfera ---
        # Esta é a fórmula matemática para uma elipsoide (uma esfera em um espaço com espaçamento desigual).
        # Para cada voxel da imagem, calculamos a sua distância ao centro da lesão,
        # levando em conta o espaçamento de cada eixo.
        distance_sq = (
            (spacing[0] * (xx - voxel_center_continuous[0]))**2 +
            (spacing[1] * (yy - voxel_center_continuous[1]))**2 +
            (spacing[2] * (zz - voxel_center_continuous[2]))**2
        )
        
        # Agora, criamos a máscara. A lógica é simples: se a distância ao quadrado de um voxel
        # for menor ou igual ao raio ao quadrado, ele está DENTRO da esfera.
        # O resultado é uma matriz 3D com 'True' (verdadeiro) para voxels dentro da esfera
        # e 'False' (falso) para os de fora.
        # O '.astype(np.uint8)' converte 'True' para 1 e 'False' para 0.
        mask_np = (distance_sq <= radius_mm**2).astype(np.uint8)

        # --- Passo 4: Converter a Matriz de Volta para uma Imagem Médica ---
        # Transforma nossa matriz NumPy (cheia de 0s e 1s) de volta em um objeto de imagem SimpleITK.
        mask_itk = sitk.GetImageFromArray(mask_np)

        # Este é um passo vital. Copiamos todas as informações espaciais (origem, espaçamento, direção)
        # da imagem original para a nossa nova máscara. Isso garante que a máscara e a imagem
        # estejam perfeitamente alinhadas no espaço 3D.
        mask_itk.CopyInformation(image_sitk)
        
        # Retorna a máscara 3D final, pronta para ser usada pelo PyRadiomics.
        return mask_itk