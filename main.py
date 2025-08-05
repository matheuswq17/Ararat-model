# Este é o arquivo que nós executamos para iniciar todo o projeto.

# Importamos a classe principal do nosso projeto, a RadiomicsPipeline, que está 
# definida no arquivo 'pipeline.py'. Ela é o nosso "gerente de produção".

from classes.pipeline import RadiomicsPipeline

# Importamos a biblioteca 'os', que nos ajuda a lidar com caminhos de arquivos 
# e pastas de uma forma que funciona em qualquer sistema operacional (Windows, Mac, Linux).
import os

# Este é o bloco de código principal que será executado quando rodarmos "python main.py".
# É o ponto de partida oficial do nosso programa.
if __name__ == "__main__":

    # ==============================================================================
    # 1. CONFIGURAÇÕES DO PROJETO
    # ==============================================================================
    # Nesta seção, definimos todas as variáveis e caminhos importantes.
    # Se alguém for usar nosso projeto, é principalmente aqui que ela precisará
    # verificar se os nomes das pastas e arquivos estão corretos.

    # Define o nome da pasta principal onde todos os nossos dados brutos estão guardados.
    # caminho para a pasta que tem os dados brutos (PROSTATEx, planilha.csv)
    DATA_FOLDER = 'data'
    
    # caminho completo para a planilha e para a pasta base das imagens
    
    # Usando a biblioteca 'os', nós montamos o caminho completo para a nossa "lista de tarefas",
    # que é a planilha com as informações das lesões.
    CSV_PATH = os.path.join(DATA_FOLDER, 'ProstateX-2-Findings-Train.csv')

    # Da mesma forma, montamos o caminho para o nosso "almoxarifado de imagens",
    # a pasta que contém as subpastas de todos os pacientes.
    IMAGES_BASE_PATH = os.path.join(DATA_FOLDER, 'PROSTATEx')
    
    # Imprime uma mensagem no terminal para o usuário saber que o processo começou.
    print("Iniciando o pipeline de extração de características radiômicas.")
    
    # ==============================================================================
    # 2. INICIALIZAÇÃO E EXECUÇÃO DO PIPELINE
    # ==============================================================================
    # Com tudo configurado, agora vamos criar e ligar a nossa "fábrica".

    # Aqui, nós criamos o nosso 'trabalhador' principal, uma instância da classe RadiomicsPipeline.
    # Nós passamos para ele todas as informações que ele precisa para trabalhar:
    pipeline = RadiomicsPipeline(
        spreadsheet_path=CSV_PATH,          # -> Onde está a planilha com a lista de lesões.
        images_base_path=IMAGES_BASE_PATH,  # -> Onde está a pasta com as imagens dos pacientes
        series_id='t2tsetra'                # -> Qual tipo de imagem procurar (a T2-weighted Axial).
                                            # -> Se quiséssemos analisar o ADC, mudaríamos aqui.
    )
    

    # Com o 'trabalhador' (pipeline) criado e configurado, damos a ordem para ele começar
    # a executar todo o processo de extração.
    # Ele vai ler a planilha, encontrar as imagens, criar as máscaras, extrair as features
    # e, ao final, nos devolver uma grande tabela (DataFrame) com todos os resultados.
    # executa a pipeline
    results_df = pipeline.run()
    
    # ==============================================================================
    # 3. EXIBIÇÃO DOS RESULTADOS
    # ==============================================================================
    # Após a conclusão do trabalho pesado, mostramos um pequeno resumo para o usuário.

    # Uma verificação de segurança: se o pipeline realmente produziu algum resultado 
    # (ou seja, se a tabela de resultados não for nula)

    if results_df is not None:
        # ...então imprimimos uma mensagem de sucesso e mostramos as 5 primeiras linhas da tabela final.
        # Isso nos dá uma confirmação visual imediata de que tudo funcionou como esperado.
        print("\nResumo dos resultados:")
        print(results_df.head())