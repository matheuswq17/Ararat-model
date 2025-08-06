from classes.pipeline import RadiomicsPipeline
import os

if __name__ == "__main__":

    # --- 1. CONFIGURAÇÕES GERAIS ---
    DATA_FOLDER = 'data'
    CSV_PATH = os.path.join(DATA_FOLDER, 'ProstateX-2-Findings-Train.csv')
    IMAGES_BASE_PATH = os.path.join(DATA_FOLDER, 'PROSTATEx')
    
    # --- 2. LISTA DE TAREFAS DE EXTRAÇÃO ---
    # Aqui definimos todas as sequências que queremos analisar.
    # Para cada uma, damos o nome da série e o nome do arquivo de saída.
    extraction_tasks = [
        {
            'series_id': 't2tsetra',
            'output_filename': 'radiomics_features_t2.csv'
        },
        {
            'series_id': 'ADC',
            'output_filename': 'radiomics_features_adc.csv'
        }
        # Se você quisesse adicionar DWI, seria só adicionar um novo dicionário aqui
    ]

    print("="*50)
    print("INICIANDO PROCESSO DE EXTRAÇÃO DE MÚLTIPLAS SÉRIES")
    print("="*50)

    # --- 3. EXECUÇÃO DO PIPELINE EM LOOP ---
    # O script agora vai rodar o pipeline uma vez para cada tarefa da lista.
    for task in extraction_tasks:
        print(f"\n--- Processando tarefa: {task['series_id']} ---")
        
        # Cria uma instância do pipeline com as configurações da tarefa atual
        pipeline = RadiomicsPipeline(
            spreadsheet_path=CSV_PATH,
            images_base_path=IMAGES_BASE_PATH,
            series_id=task['series_id'],
            output_filename=task['output_filename']
        )
        
        # Executa o pipeline para esta tarefa
        pipeline.run()

    print("\n" + "="*50)
    print("TODAS AS TAREFAS DE EXTRAÇÃO FORAM CONCLUÍDAS!")
    print("="*50)