from classes.pipeline import RadiomicsPipeline
import os

if __name__ == "__main__":
    # caminho para a pasta que tem os dados brutos (PROSTATEx, planilha.csv)
    DATA_FOLDER = 'data'
    
    # caminho completo para a planilha e para a pasta base das imagens
    CSV_PATH = os.path.join(DATA_FOLDER, 'ProstateX-2-Findings-Train.csv')
    IMAGES_BASE_PATH = os.path.join(DATA_FOLDER, 'PROSTATEx')
    
    print("Iniciando o pipeline de extração de características radiômicas.")
    
    # cria uma instância da pipeline
    pipeline = RadiomicsPipeline(
        spreadsheet_path=CSV_PATH,
        images_base_path=IMAGES_BASE_PATH,
        series_id='t2tsetra'  # altere se necessário
    )
    
    # executa a pipeline
    results_df = pipeline.run()
    
    if results_df is not None:
        print("\nResumo dos resultados:")
        print(results_df.head())