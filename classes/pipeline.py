import pandas as pd
import SimpleITK as sitk
import os
from radiomics.featureextractor import RadiomicsFeatureExtractor
from classes.segmentation import Segmentation

class RadiomicsPipeline():
    # adicionado 'output_filename' como um parâmetro aqui
    def __init__(self, spreadsheet_path, images_base_path, series_id='t2_tse_tra', output_filename="radiomics_features.csv"):
        self.spreadsheet_path = spreadsheet_path
        self.images_base_path = images_base_path
        self.series_id = series_id
        # salva o nome do arquivo de saída
        self.output_filename = output_filename
        self.extractor = RadiomicsFeatureExtractor()
        print(f"RadiomicsPipeline inicializado para a série '{self.series_id}'.")

    def _find_dicom_series(self, patient_path):
        for root, dirs, files in os.walk(patient_path):
            # adicionada uma pequena lógica para lidar com nomes de série ADC que as vezes são apenas números
            series_to_find = self.series_id.lower()
            for d in dirs:
                dir_lower = d.lower()
                # A condição agora é mais flexível
                if (series_to_find in dir_lower or series_to_find.replace(" ", "") in dir_lower.replace(" ", "")) and len(os.listdir(os.path.join(root, d))) > 10:
                    print(f"  > Série encontrada: {d}")
                    return os.path.join(root, d)
        raise FileNotFoundError(f"Nenhuma série correspondente a '{self.series_id}' encontrada em {patient_path}")

    def run(self):
        try:
            df_lesoes = pd.read_csv(self.spreadsheet_path)
            df_lesoes = df_lesoes.rename(columns={'ProxID': 'PatientID', 'pos': 'WorldCoordinates', 'fid': 'FindingID'})
            print(f"Planilha de lesões carregada com {len(df_lesoes)} lesões.")
        except FileNotFoundError:
            print(f"ERRO CRÍTICO: Planilha de lesões não encontrada em '{self.spreadsheet_path}'.")
            return

        all_results = []
        
        for index, row in df_lesoes.iterrows():
            patient_id = row['PatientID']
            finding_id = row['FindingID']
            print(f"\n[Processando] Paciente: {patient_id}, Lesão: {finding_id}")
            
            try:
                coords_mm = [float(c) for c in row['WorldCoordinates'].split()]
                patient_path = os.path.join(self.images_base_path, patient_id)
                series_path = self._find_dicom_series(patient_path)
                
                image_sitk = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(series_path))
                mask_sitk = Segmentation.create_image_mask(image_sitk, coords_mm, radius_mm=5)
                features = self.extractor.execute(image_sitk, mask_sitk)
                
                features['PatientID'] = patient_id
                features['FindingID'] = finding_id
                
                if 't2tsetra' in self.series_id:
                    for col in ['ggg', 'zone', 'ClinSig']: # colunas adicionais
                        if col in row and pd.notna(row[col]):
                            features[col] = row[col]

                all_results.append(features)
                print(f"  [SUCESSO] Features extraídas para a lesão {finding_id}.")
            except Exception as e:
                print(f"  !!!!!!!! [FALHA] ao processar {patient_id}, Lesão {finding_id}. Erro: {e}")
        
        if not all_results:
            print("\nNenhuma lesão foi processada com sucesso.")
            return

        df_final = pd.DataFrame(all_results)
        df_final.to_csv(self.output_filename, index=False)
        print(f"\n\nPIPELINE PARA '{self.series_id}' CONCLUÍDO! Arquivo '{self.output_filename}' salvo com {len(df_final)} linhas.")
        return df_final