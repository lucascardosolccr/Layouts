import os
import io
import sys
import json
import logging
import zipfile
import shutil
import tempfile
import pandas as pd
import numpy as np
import unicodedata
import re
import traceback
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sweetviz as sv
import xlsxwriter
from tqdm import tqdm
import warnings

# Ignorar warnings não críticos
warnings.filterwarnings('ignore')

# Configuração da Página Streamlit
st.set_page_config(
    page_title="Inteligência Analítica de Salas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CONFIGURAÇÃO DE LOGS (Adaptada para UI e Arquivo)
# ==========================================
log_stream = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analise_log.txt"), 
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_stream)
    ]
)
logger = logging.getLogger("InteligenciaAnalitica")

# ==========================================
# NÚCLEO DA APLICAÇÃO (Preservado e Evoluído)
# ==========================================

class DataLoaderAndCleaner:
    """Responsável pela ingestão, detecção de tipos e higienização avançada dos dados."""
    def __init__(self, filepath, fix_encoding=True):
        self.filepath = filepath
        self.df = None
        self.quality_report = {}
        self.fix_encoding = fix_encoding 

    def load_data(self):
        logger.info(f"Ingerindo conjunto de dados a partir de: {self.filepath}")
        try:
            self.df = pd.read_excel(self.filepath)
        except Exception as e:
            logger.error(f"Falha crítica na ingestão do arquivo: {e}")
            raise

    def decode_sas_windows_artifacts(self, text):
        """Desfaz artefatos do SAS (Western/cp1252) e códigos Hexadecimais do Excel."""
        if pd.isna(text): 
            return 'NAO_INFORMADO'
            
        text = str(text)
        
        # Limpeza de injeções XML do Excel
        text = re.sub(r'_x([0-9a-fA-F]{4})_', lambda m: chr(int(m.group(1), 16)), text, flags=re.IGNORECASE)
        
        try:
            # Reversão de distorção de Bytes SAS (Windows-1252 para UTF-8)
            if 'Ã' in text: 
                text = text.encode('cp1252').decode('utf-8')
        except: 
            pass

        # Correção Mojibake Manual de Backup
        mojibake_map = {
            'Ãƒ': 'Ã', 'Ã‡': 'Ç', 'Ã ': 'Á', 'Ã‰': 'É', 'Ã ': 'Í', 
            'Ã“': 'Ó', 'Ãš': 'Ú', 'Ã‚': 'Â', 'ÃŠ': 'Ê', 'Ã”': 'Ô', 
            'Ã•': 'Õ', 'Ã€': 'À', 'Ã£': 'ã', 'Ã§': 'ç', 'Ã¡': 'á', 
            'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã¢': 'â', 
            'Ãª': 'ê', 'Ã´': 'ô', 'Ãµ': 'õ', 'Ã ': 'à'
        }
        for errado, certo in mojibake_map.items():
            text = text.replace(errado, certo)
            
        return text

    def clean_text_legacy(self, text):
        """[MÉTODO RESTAURADO E MANTIDO - NÃO REGRESSÃO]"""
        correcoes = {
            'EDUCA??O': 'EDUCACAO', 
            'ROS?RIO': 'ROSARIO', 
            'PR?DIO': 'PREDIO'
        }
        for errado, certo in correcoes.items():
            text = text.replace(errado, certo)
            
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^A-Z0-9\s\-\.\/]', '', text)
        return text

    def clean_text_pipeline(self, text):
        """Pipeline central de sanitização não destrutiva."""
        cleaned = self.decode_sas_windows_artifacts(text)
        cleaned = cleaned.upper().strip()
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned in ['NAN', 'NONE', 'NULL', '']: 
            return 'NAO_INFORMADO'
            
        return cleaned

    def clean_and_validate(self):
        logger.info("Iniciando pipeline de sanitização de dados e desencriptação SAS/XML...")
        self.quality_report['total_linhas_iniciais'] = len(self.df)
        self.quality_report['duplicadas_encontradas'] = int(self.df.duplicated().sum())
        self.quality_report['nulos_por_coluna'] = self.df.isnull().sum().to_dict()
        
        self.df.drop_duplicates(inplace=True)
        
        # ---------------------------------------------------------
        # BLINDAGEM CONTRA TYPE MISMATCH (Numpy Int64 vs String)
        # ---------------------------------------------------------
        # Força as colunas identificadoras a serem tratadas como texto, 
        # impedindo falhas em operações de agrupamento ou plotagem.
        categorical_identifiers = ['NO_LOCAL', 'SG_UF', 'NO_SALA', 'CO_ENTIDADE']
        for col in categorical_identifiers:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        # ---------------------------------------------------------

        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].apply(self.clean_text_pipeline)
            
        cap_col = 'QT_CAPACIDADE_MAXIMA_SALA'
        if cap_col in self.df.columns:
            self.df[cap_col] = pd.to_numeric(self.df[cap_col], errors='coerce')
            self.quality_report['capacidades_invalidas'] = int(self.df[cap_col].isnull().sum())
            self.df[cap_col].fillna(self.df[cap_col].median(), inplace=True)
        
        self.quality_report['total_linhas_finais'] = len(self.df)
        return self.df, self.quality_report


class StatisticalAnalyzer:
    """Módulo de estatística inferencial, descritiva e categorização baseada em dados."""
    def __init__(self, df):
        self.df = df
        self.cap_col = 'QT_CAPACIDADE_MAXIMA_SALA'

    def apply_capacity_bins(self):
        """Categoriza as salas calculando dinamicamente os limites estatísticos (Quartis)."""
        logger.info("Processando segmentação analítica da FAIXA_CAPACIDADE...")
        if self.cap_col not in self.df.columns: 
            return None, {}
        
        s = self.df[self.cap_col]
        q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
        media = s.mean()
        std = s.std()
        limite_superior = media + (2 * std)
        
        condicoes = [
            (s <= q25),
            (s > q25) & (s <= q50),
            (s > q50) & (s <= q75),
            (s > q75) & (s <= limite_superior),
            (s > limite_superior)
        ]
        categorias = [
            '1. Micro Sala', 
            '2. Sala Pequena', 
            '3. Sala Média', 
            '4. Sala Grande', 
            '5. Gigante / Auditório'
        ]
        
        self.df['FAIXA_CAPACIDADE'] = np.select(condicoes, categorias, default='Não Classificado')
        faixas_df = self.df['FAIXA_CAPACIDADE'].value_counts().reset_index()
        
        cutoffs = {
            'q25': int(q25) if pd.notnull(q25) else 0,
            'q50': int(q50) if pd.notnull(q50) else 0,
            'q75': int(q75) if pd.notnull(q75) else 0,
            'limite_sup': int(limite_superior) if pd.notnull(limite_superior) else 0
        }
        return faixas_df, cutoffs

    def get_general_stats(self):
        logger.info("Computando medidas de tendência central e dispersão...")
        s = self.df[self.cap_col]
        stat_norm, p_norm = stats.normaltest(s.dropna())
        is_normal = "Sim" if p_norm > 0.05 else "Não (Distribuição Assimétrica)"

        stats_dict = {
            'N (Tamanho da Amostra / Salas)': len(self.df),
            'Locais Independentes': self.df['NO_LOCAL'].nunique() if 'NO_LOCAL' in self.df else 0,
            'Dispersão Geográfica (UFs)': self.df['SG_UF'].nunique() if 'SG_UF' in self.df else 0,
            'Capacidade Total Instalada': s.sum(),
            'Média Aritmética (\u03BC)': s.mean(),
            'Mediana (Ponto de Separação)': s.median(),
            'Moda (Valor de Maior Densidade)': s.mode()[0] if not s.mode().empty else None,
            'Desvio Padrão (\u03C3)': s.std(),
            'Coeficiente de Variação (CV %)': (s.std() / s.mean()) * 100 if s.mean() != 0 else 0,
            'Mínimo Global': s.min(),
            'Máximo Global': s.max(),
            'Amplitude Total': s.max() - s.min(),
            'Primeiro Quartil (Q1 - 25%)': s.quantile(0.25),
            'Terceiro Quartil (Q3 - 75%)': s.quantile(0.75),
            'Assimetria (Skewness)': s.skew(),
            'Curtose (Achatamento da Curva)': s.kurtosis(),
            'Aderência à Distribuição Normal?': is_normal
        }
        return pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Valor Estimado']).round(2)

    def detect_outliers(self):
        logger.info("Processando detecção paramétrica de Outliers...")
        s = self.df[self.cap_col]
        z_scores = np.abs(stats.zscore(s))
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_df = self.df[(z_scores > 3) | (s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))].copy()
        outliers_df['CLASSIFICACAO_ESTATISTICA'] = np.where(
            outliers_df[self.cap_col] > s.mean(), 
            'Outlier Superior (Risco de Sobreposição / Auditório Gigante)', 
            'Outlier Inferior (Risco de Ineficiência Logística)'
        )
        return outliers_df

    def group_analysis(self, group_by_cols):
        grouped = self.df.groupby(group_by_cols).agg(
            QTD_SALAS=(self.cap_col, 'count'),
            CAPACIDADE_TOTAL=(self.cap_col, 'sum'),
            CAPACIDADE_MEDIA=(self.cap_col, 'mean'),
            CAPACIDADE_MAXIMA=(self.cap_col, 'max')
        ).reset_index()
        
        total_nacional = grouped['CAPACIDADE_TOTAL'].sum()
        grouped['REPRESENTATIVIDADE (%)'] = (grouped['CAPACIDADE_TOTAL'] / total_nacional) * 100
        grouped['RANKING_VOLUMETRICO'] = grouped['CAPACIDADE_TOTAL'].rank(ascending=False, method='min')
        return grouped.sort_values('CAPACIDADE_TOTAL', ascending=False).round(2)

    def get_top_locais(self, n=20):
        if 'NO_LOCAL' not in self.df.columns: 
            return pd.DataFrame()
        locais = self.group_analysis(['NO_LOCAL', 'SG_UF'] if 'SG_UF' in self.df.columns else ['NO_LOCAL'])
        return locais.head(n)

    def pareto_analysis(self, col):
        pareto_df = self.df.groupby(col)[self.cap_col].sum().sort_values(ascending=False).reset_index()
        pareto_df['FREQUENCIA_RELATIVA (%)'] = (pareto_df[self.cap_col] / pareto_df[self.cap_col].sum()) * 100
        pareto_df['FREQUENCIA_ACUMULADA (%)'] = pareto_df['FREQUENCIA_RELATIVA (%)'].cumsum()
        return pareto_df.round(2)


class MachineLearningEngine:
    """Implementação do algoritmo K-Means para segmentação comportamental não-supervisionada."""
    def __init__(self, df):
        self.df = df
        
    def cluster_locations(self):
        logger.info("Treinando modelo K-Means para Clusterização de IA...")
        if 'NO_LOCAL' not in self.df.columns: 
            return pd.DataFrame()
            
        locais = self.df.groupby('NO_LOCAL').agg(
            QTD_SALAS=('QT_CAPACIDADE_MAXIMA_SALA', 'count'),
            CAPACIDADE_MEDIA=('QT_CAPACIDADE_MAXIMA_SALA', 'mean'),
            CAPACIDADE_TOTAL=('QT_CAPACIDADE_MAXIMA_SALA', 'sum')
        ).reset_index()
        
        X = locais[['QTD_SALAS', 'CAPACIDADE_MEDIA', 'CAPACIDADE_TOTAL']]
        X_scaled = StandardScaler().fit_transform(X)
        
        # Fit do K-Means com 4 clusters fixos para garantir coerência de negócio
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        locais['ID_CLUSTER'] = kmeans.fit_predict(X_scaled)
        
        cluster_means = locais.groupby('ID_CLUSTER')['CAPACIDADE_TOTAL'].mean().sort_values()
        labels = {
            cluster_means.index[0]: 'C1: Operação Base (Pulverizados/Menores)',
            cluster_means.index[1]: 'C2: Polos Intermediários (Comportamento Padrão)',
            cluster_means.index[2]: 'C3: Centros Estratégicos (Alta Concentração)',
            cluster_means.index[3]: 'C4: Super-Polos Logísticos (Risco Elevado/Massivos)'
        }
        locais['PERFIL_COMPORTAMENTAL_IA'] = locais['ID_CLUSTER'].map(labels)
        
        # Injeta o resultado do K-Means de volta na base principal
        self.df = self.df.merge(locais[['NO_LOCAL', 'PERFIL_COMPORTAMENTAL_IA']], on='NO_LOCAL', how='left')
        return locais.sort_values('CAPACIDADE_TOTAL', ascending=False).round(2)


class VisualizerAndExporter:
    """Responsável por TODA a interface BI Interativa (SPA), Plotly Históricos, HTMLs, TXT e Excel."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.graficos_dir = f"{output_dir}/graficos"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.graficos_dir, exist_ok=True)
        self.generated_figs = {}

    def plot_and_save(self, fig, filename):
        fig.write_html(f"{self.graficos_dir}/{filename}.html", include_plotlyjs='cdn')
        try:
            fig.write_image(f"{self.graficos_dir}/{filename}.png", scale=3)
            fig.write_image(f"{self.graficos_dir}/{filename}.svg")
        except Exception:
            pass 
        self.generated_figs[filename] = fig

    def generate_charts(self, df, uf_stats, pareto_uf, top_locais, faixas_df, clusters_df):
        logger.info("Gerando os 13 Paineis de Visualização Científica Plotly...")
        
        # 1. KPIs
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Indicator(mode="number", value=df['QT_CAPACIDADE_MAXIMA_SALA'].sum(), title={"text": "Capacidade Nacional"}, domain={'row': 0, 'column': 0}))
        fig_kpi.add_trace(go.Indicator(mode="number", value=df['NO_LOCAL'].nunique(), title={"text": "Locais"}, domain={'row': 0, 'column': 1}))
        fig_kpi.add_trace(go.Indicator(mode="number", value=df['QT_CAPACIDADE_MAXIMA_SALA'].mean(), title={"text": "Média"}, domain={'row': 0, 'column': 2}))
        fig_kpi.update_layout(grid={'rows': 1, 'columns': 3, 'pattern': "independent"})
        self.plot_and_save(fig_kpi, '1_KPIs_Executivos')

        # 2. Histograma
        fig_hist = px.histogram(df, x="QT_CAPACIDADE_MAXIMA_SALA", nbins=50, marginal="box", title="Densidade de Capacidades")
        self.plot_and_save(fig_hist, '2_Histograma_Distribuicao')

        # 3. Treemap
        if 'SG_UF' in df.columns and 'NO_LOCAL' in df.columns:
            df_tree = df.copy()
            # Double safety casting text to bypass any Plotly numerical sort exceptions
            df_tree['NO_LOCAL'] = df_tree['NO_LOCAL'].astype(str)
            df_tree['SG_UF'] = df_tree['SG_UF'].astype(str)
            
            limite = df_tree['QT_CAPACIDADE_MAXIMA_SALA'].quantile(0.70)
            df_tree.loc[df_tree['QT_CAPACIDADE_MAXIMA_SALA'] < limite, 'NO_LOCAL'] = 'DEMAIS POLOS'
            fig_tree = px.treemap(df_tree, path=[px.Constant("Brasil"), 'SG_UF', 'NO_LOCAL'], values='QT_CAPACIDADE_MAXIMA_SALA', title='Treemap: Hierarquia Logística')
            self.plot_and_save(fig_tree, '3_Treemap_Concentracao')

        # 4. Violino
        fig_vio = px.violin(df, y="QT_CAPACIDADE_MAXIMA_SALA", x="SG_UF", box=True, points="outliers", title="Violin Plot: Assimetria Regional")
        self.plot_and_save(fig_vio, '4_Violino_Densidade')

        # 5. Pareto
        fig_par = go.Figure([
            go.Bar(x=pareto_uf['SG_UF'], y=pareto_uf['QT_CAPACIDADE_MAXIMA_SALA'], name='Bruta'),
            go.Scatter(x=pareto_uf['SG_UF'], y=pareto_uf['FREQUENCIA_ACUMULADA (%)'], name='Acumulada %', yaxis='y2')
        ])
        fig_par.update_layout(title='Teorema de Pareto (80/20)', yaxis2=dict(overlaying='y', side='right', range=[0, 100]))
        self.plot_and_save(fig_par, '5_Pareto_UF')

        # 6. Top 20
        fig_top = px.bar(top_locais.sort_values('CAPACIDADE_TOTAL', ascending=True), x='CAPACIDADE_TOTAL', y='NO_LOCAL', orientation='h', color='SG_UF', title='Top 20 Super-Polos')
        self.plot_and_save(fig_top, '6_Top_20_Locais')
        
        # 7. Faixas
        if faixas_df is not None:
            faixas_df.columns = ['FAIXA_CAPACIDADE', 'VOLUMETRIA']
            fig_faixa = px.bar(faixas_df.sort_values('FAIXA_CAPACIDADE'), x='FAIXA_CAPACIDADE', y='VOLUMETRIA', title='Distribuição por Faixa', color='FAIXA_CAPACIDADE', text_auto=True)
            self.plot_and_save(fig_faixa, '7_Faixas_Capacidade')

        # 8. IA Clusters
        if not clusters_df.empty:
            cluster_counts = clusters_df['PERFIL_COMPORTAMENTAL_IA'].value_counts().reset_index()
            cluster_counts.columns = ['PERFIL_COMPORTAMENTAL_IA', 'QTD_LOCAIS']
            fig_ia = px.pie(cluster_counts, values='QTD_LOCAIS', names='PERFIL_COMPORTAMENTAL_IA', title='K-Means: Perfis de IA', hole=0.4)
            self.plot_and_save(fig_ia, '8_Clusters_IA_Distribuicao')

        # 9. Boxplot Faixas
        if 'FAIXA_CAPACIDADE' in df.columns:
            fig_box_faixa = px.box(df, x="FAIXA_CAPACIDADE", y="QT_CAPACIDADE_MAXIMA_SALA", color="FAIXA_CAPACIDADE", title="Dispersão Interna por Faixa", category_orders={"FAIXA_CAPACIDADE": ['1. Micro Sala', '2. Sala Pequena', '3. Sala Média', '4. Sala Grande', '5. Gigante / Auditório']})
            self.plot_and_save(fig_box_faixa, '9_Boxplot_Faixas')

        # 10. Waterfall
        if not uf_stats.empty:
            uf_wf = uf_stats.head(10).copy()
            outros_val = uf_stats.iloc[10:]['CAPACIDADE_TOTAL'].sum() if len(uf_stats) > 10 else 0
            
            # Type casting blindado na estrutura para ordenação correta do eixo X no Plotly
            wf_names = [str(x) for x in uf_wf['SG_UF']] + ['DEMAIS ESTADOS']
            wf_vals = list(uf_wf['CAPACIDADE_TOTAL']) + [outros_val]
            
            fig_wf = go.Figure(go.Waterfall(name="Acumulado", orientation="v", measure=["relative"]*10 + ["total"],
                x=wf_names, y=wf_vals, textposition="outside", text=[f"{v:,.0f}" for v in wf_vals], connector={"line":{"color":"rgb(63, 63, 63)"}}))
            fig_wf.update_layout(title="Waterfall: Composição Cumulativa Nacional")
            self.plot_and_save(fig_wf, '10_Waterfall_UFs')

        # 11. Scatter Dispersão
        if not clusters_df.empty:
            fig_scatter = px.scatter(clusters_df, x="QTD_SALAS", y="CAPACIDADE_MEDIA", color="PERFIL_COMPORTAMENTAL_IA", size="CAPACIDADE_TOTAL", hover_name="NO_LOCAL", title="Dispersão: Qtd vs Cap. Média")
            self.plot_and_save(fig_scatter, '11_Dispersao_Salas_vs_Capacidade')

        # 12. Heatmap de Correlação
        if not clusters_df.empty:
            corr_cols = ['QTD_SALAS', 'CAPACIDADE_MEDIA', 'CAPACIDADE_TOTAL']
            corr_matrix = clusters_df[corr_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, title="Matriz de Correlação de Pearson", aspect="auto", color_continuous_scale='RdBu_r')
            self.plot_and_save(fig_corr, '12_Heatmap_Correlacao')

        # 13. ECDF Acumulada
        df_sorted = df.sort_values('QT_CAPACIDADE_MAXIMA_SALA')
        df_sorted['Acumulado'] = np.arange(1, len(df_sorted)+1) / len(df_sorted)
        fig_ecdf = px.line(df_sorted, x='QT_CAPACIDADE_MAXIMA_SALA', y='Acumulado', title='Curva de Distribuição Cumulativa Empírica (ECDF)')
        self.plot_and_save(fig_ecdf, '13_ECDF_Acumulada')

    def generate_html_report(self, df):
        logger.info("Iniciando geração do Relatório Exploratório Sweetviz...")
        try:
            report = sv.analyze(df)
            report.show_html(filepath=f"{self.output_dir}/1_Sweetviz_Original_Tecnico.html", open_browser=False)
        except Exception as e:
            logger.error(f"Omissão Graciosa Sweetviz: {e}")

    def create_custom_portuguese_dashboard(self, df, cutoffs, geral_stats):
        """Gera o Web App SPA (HTML/JS) com Explainable Analytics e Cross-Filtering Restaurado."""
        logger.info("Compilando Plataforma de BI Interativa (Single Page Application)...")
        
        df_json = df.to_dict(orient='records')
        json_str = json.dumps(df_json, ensure_ascii=False)
        
        media_val = geral_stats.loc['Média Aritmética (\u03BC)', 'Valor Estimado']
        mediana_val = geral_stats.loc['Mediana (Ponto de Separação)', 'Valor Estimado']
        skew_val = geral_stats.loc['Assimetria (Skewness)', 'Valor Estimado']
        cv_val = geral_stats.loc['Coeficiente de Variação (CV %)', 'Valor Estimado']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Plataforma BI: Analítica Estratégica</title>
            
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
            <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
            <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
            
            <style>
                :root {{ --primary: #0f4c81; --secondary: #2980b9; --citizen: #e67e22; --tech: #8e44ad; --bg: #f4f7f6; --danger: #c0392b; --success: #27ae60; }}
                body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; background-color: var(--bg); color: #333; margin: 0; padding: 0; display: flex; }}
                
                /* Explainable Analytics Tooltips */
                .tooltip-icon {{ cursor: help; background: var(--secondary); color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-block; text-align: center; font-size: 13px; font-weight: bold; line-height: 20px; margin-left: 8px; position: relative; font-family: monospace; }}
                .tooltip-icon .tooltip-text {{ visibility: hidden; width: 350px; background-color: #2c3e50; color: #fff; text-align: left; padding: 15px; border-radius: 6px; position: absolute; z-index: 9999; bottom: 130%; left: 50%; transform: translateX(-50%); font-weight: normal; font-size: 13px; line-height: 1.5; box-shadow: 0 4px 15px rgba(0,0,0,0.4); opacity: 0; transition: opacity 0.3s; pointer-events: none; border-left: 4px solid var(--citizen); }}
                .tooltip-icon .tooltip-text::after {{ content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #2c3e50 transparent transparent transparent; }}
                .tooltip-icon:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
                .tooltip-formula {{ color: #f1c40f; font-family: monospace; display: block; margin-top: 5px; margin-bottom: 5px; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 3px; }}

                /* Sidebar */
                .sidebar {{ width: 300px; background: #fff; height: 100vh; position: fixed; padding: 25px; box-shadow: 2px 0 10px rgba(0,0,0,0.05); overflow-y: auto; z-index: 1000; border-right: 1px solid #ddd; box-sizing: border-box; }}
                .sidebar h2 {{ color: var(--primary); font-size: 1.2rem; border-bottom: 2px solid var(--secondary); padding-bottom: 10px; margin-top: 0; }}
                .filter-group {{ margin-bottom: 20px; }}
                .filter-group label {{ font-weight: 600; display: block; margin-bottom: 8px; font-size: 0.9em; color: #555; }}
                .filter-group select {{ width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-size: 0.95em; outline: none; }}
                .btn-clear {{ background-color: var(--danger); color: white; border: none; padding: 12px; width: 100%; border-radius: 5px; cursor: pointer; font-weight: bold; margin-top: 10px; transition: background 0.3s; text-transform: uppercase; font-size: 0.85em; }}
                .btn-clear:hover {{ background-color: #a93226; }}
                
                /* Main */
                .main-content {{ margin-left: 300px; padding: 40px; flex-grow: 1; max-width: 1400px; box-sizing: border-box; }}
                h1, h2, h3 {{ color: var(--primary); }}
                h1 {{ border-bottom: 3px solid var(--secondary); padding-bottom: 10px; font-size: 2.2rem; }}
                
                /* Insights / Perfil */
                #insight-box {{ display: none; background: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; margin-bottom: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                #perfil-analitico {{ background: #e8f4f8; padding: 25px; border-radius: 8px; border-left: 5px solid var(--secondary); margin-bottom: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: all 0.3s ease; }}
                .perfil-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
                .perfil-item {{ background: #fff; padding: 15px; border-radius: 5px; border: 1px solid #dcdde1; }}
                .perfil-item span {{ font-size: 1.4em; font-weight: bold; color: var(--secondary); display:block; }}

                /* KPIs */
                .kpi-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .kpi-card {{ background: white; padding: 25px 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center; border-bottom: 4px solid var(--secondary); transition: transform 0.2s; }}
                .kpi-card:hover {{ transform: translateY(-5px); }}
                .kpi-value {{ font-size: 2.5em; font-weight: 800; color: var(--primary); margin-bottom: 5px; }}
                .kpi-label {{ font-size: 0.85em; color: #7f8c8d; text-transform: uppercase; font-weight: 600; display: flex; justify-content: center; align-items: center; }}
                
                /* Cartões Gráficos */
                .card {{ background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 40px; }}
                .chart-explainer {{ display: flex; gap: 20px; margin-bottom: 25px; flex-wrap: wrap; background: #fafafa; padding: 20px; border-radius: 6px; border: 1px solid #eee; }}
                .explainer-col {{ flex: 1; min-width: 300px; padding: 0 15px; }}
                .explainer-col.citizen {{ border-left: 4px solid var(--citizen); }}
                .explainer-col.tech {{ border-left: 4px solid var(--tech); }}
                .explainer-col h4 {{ margin-top: 0; font-size: 1em; text-transform: uppercase; letter-spacing: 0.5px; }}
                .plotly-chart {{ width: 100%; height: 500px; }}
                iframe {{ width: 100%; height: 550px; border: none; margin-top: 15px; }}
                .data-table-container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 50px; }}
                
                .help-box {{ background: #34495e; color: #fff; padding: 20px; border-radius: 8px; margin-top: 30px; font-size: 0.9em; }}
                .help-box h3 {{ color: #f1c40f; border-bottom: 1px solid #7f8c8d; padding-bottom: 10px; margin-top: 0; }}
            </style>
        </head>
        <body>
            <!-- BARRA LATERAL (SPA) -->
            <div class="sidebar">
                <h2>Painel Analítico (Filtros)</h2>
                <div class="filter-group">
                    <label>📍 Filtrar por Estado (UF):</label>
                    <select id="filterUF" onchange="updateDashboard()"><option value="ALL">Brasil (Todas as UFs)</option></select>
                </div>
                <div class="filter-group">
                    <label>📏 Faixa de Capacidade:</label>
                    <select id="filterFaixa" onchange="updateDashboard()"><option value="ALL">Todas as Faixas</option></select>
                </div>
                <div class="filter-group">
                    <label>🤖 Perfil de IA (Clusters):</label>
                    <select id="filterCluster" onchange="updateDashboard()"><option value="ALL">Todos os Perfis</option></select>
                </div>
                <button class="btn-clear" onclick="resetFilters()">Remover Filtros</button>
                
                <div class="help-box">
                    <h3>Como Usar este Painel?</h3>
                    <ul style="padding-left: 15px; line-height: 1.5;">
                        <li><strong>Interatividade Cruzada:</strong> Clique em uma barra no gráfico de Pareto ou numa fatia da Pizza e o relatório inteiro será filtrado para aquela seleção!</li>
                        <li><strong>O que é o "i" ?</strong> Passe o mouse sobre as bolinhas azuis <code>(i)</code> para ver a fórmula matemática exata e o parecer estatístico.</li>
                        <li><strong>Data Grid (Tabela):</strong> A tabela reflete só o que você filtrou. Você pode buscar o nome de escolas específicas nela.</li>
                    </ul>
                </div>
                
                <hr style="margin-top: 30px; border-color: #eee;">
                <h3>Índice Rápido</h3>
                <ul style="list-style: none; padding-left: 0; font-size: 0.9em; line-height: 1.8;">
                    <li><a href="#kpis" style="color:var(--secondary); text-decoration:none;">1. KPIs em Tempo Real</a></li>
                    <li><a href="#interativos" style="color:var(--secondary); text-decoration:none;">2. Painéis Dinâmicos (JS)</a></li>
                    <li><a href="#estaticos" style="color:var(--secondary); text-decoration:none;">3. Modelos Globais e Estáticos</a></li>
                    <li><a href="#tabela" style="color:var(--secondary); text-decoration:none;">4. Tabela Fato de Dados</a></li>
                </ul>
            </div>
            
            <!-- CONTEÚDO PRINCIPAL -->
            <div class="main-content">
                <h1>PLATAFORMA DE INTELIGÊNCIA LOGÍSTICA E PREDIÇÃO <div class="tooltip-icon">i<span class="tooltip-text"><strong>O que é este ambiente?</strong><br>Aplicação Single Page (SPA) baseada em Python + JavaScript. Todo o cruzamento de dados, K-Means e métricas rodam nativamente no seu navegador, sem servidores.</span></div></h1>
                
                <!-- ALERTA AUTOMÁTICO -->
                <div id="insight-box">
                    <h3 style="margin-top: 0; color: #b97a00; font-size: 1.2rem;">⚠️ Inteligência Analítica: Padrão Anômalo Detectado</h3>
                    <p id="insight-text"></p>
                </div>

                <!-- PERFIL ANALÍTICO DA SELEÇÃO -->
                <div id="perfil-analitico">
                    <h3 style="margin-top:0;">Estatística Descritiva Parcial (Seleção Atual) <div class="tooltip-icon">i<span class="tooltip-text"><strong>Transparência de Cálculo:</strong><br>O motor JS recalcula Mediana e Desvio iterando apenas nas linhas que atendem aos filtros ativos da barra lateral.</span></div></h3>
                    <p id="perfil-narrativa">Visualizando o cenário Nacional completo.</p>
                    <div class="perfil-grid">
                        <div class="perfil-item"><strong>Representatividade</strong><span id="pa-rep">100%</span></div>
                        <div class="perfil-item"><strong>Mediana Atual</strong><span id="pa-med">0</span></div>
                        <div class="perfil-item"><strong>Desvio Padrão</strong><span id="pa-std">0</span></div>
                        <div class="perfil-item"><strong>Assimetria (Skew)</strong><span id="pa-skew">0</span></div>
                    </div>
                </div>

                <h2 id="kpis" style="display:flex; align-items:center;">1. Indicadores Chave de Desempenho (KPIs) <div class="tooltip-icon">i<span class="tooltip-text"><strong>KPIs Dinâmicos</strong><br>Atualizados via Callbacks do Plotly e DataTables.<br><span class="tooltip-formula">Agregação: SOMA(Cap), COUNT(Salas)</span></span></div></h2>
                
                <div class="kpi-container">
                    <div class="kpi-card"><div class="kpi-value" id="kpiCapacidade">0</div><div class="kpi-label">Capacidade Total <div class="tooltip-icon">i<span class="tooltip-text">Lotação máxima teórica.</span></div></div></div>
                    <div class="kpi-card"><div class="kpi-value" id="kpiSalas">0</div><div class="kpi-label">Qtd. de Salas <div class="tooltip-icon">i<span class="tooltip-text">Nº de Ambientes Independentes.</span></div></div></div>
                    <div class="kpi-card"><div class="kpi-value" id="kpiLocais">0</div><div class="kpi-label">Prédios Únicos <div class="tooltip-icon">i<span class="tooltip-text">Indica a capilaridade estrutural (Endereços Físicos).</span></div></div></div>
                    <div class="kpi-card"><div class="kpi-value" id="kpiMedia">0</div><div class="kpi-label">Média por Sala <div class="tooltip-icon">i<span class="tooltip-text">Se Assimetria > 1, este valor está inflado.</span></div></div></div>
                </div>

                <h2 id="interativos" style="border-bottom: 2px solid #ccc; padding-bottom: 10px;">2. PAINEL DE EXPLORAÇÃO VISUAL (INTERATIVO JS)</h2>

                <!-- CHART 1: PARETO INTERATIVO -->
                <div class="card">
                    <h3>
                        A. Eficiência Operacional (Pareto) e Risco Estrutural
                        <div class="tooltip-icon">i
                            <span class="tooltip-text">
                                <strong>Princípio de Pareto (80/20)</strong><br>
                                <span class="tooltip-formula">Eixo X: UF em ordem decrescente.</span>
                                <span class="tooltip-formula">Linha Vermelha: Freq. Acumulada.</span><br>
                                DICA: CLIQUE na barra azul para filtrar todo o Dashboard.
                            </span>
                        </div>
                    </h3>
                    <div class="chart-explainer">
                        <div class="explainer-col citizen">
                            <h4 style="color:var(--citizen)">Gestão de Operação:</h4>
                            Siga a linha vermelha. Ao cruzar a marca de 80%, você descobre quais Estados sustentam todo o projeto. Aplique o dinheiro de segurança apenas neles.
                        </div>
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Validação Científica:</h4>
                            Curva de cumulatividade logarítmica recalculada em tempo real (JS Engine). Confirma ineficiência em alocar o mesmo recurso para a Cauda Longa.
                        </div>
                    </div>
                    <div id="chartPareto" class="plotly-chart"></div>
                </div>

                <!-- CHART 2: K-MEANS INTERATIVO -->
                <div class="card">
                    <h3>
                        B. K-Means: A Visão Paramétrica da Inteligência Artificial
                        <div class="tooltip-icon">i
                            <span class="tooltip-text">
                                <strong>K-Means Clustering</strong><br>
                                <span class="tooltip-formula">Features: [QTD_SALAS, CAPACIDADE_MEDIA, CAPACIDADE_TOTAL]</span>
                                DICA: CLIQUE numa fatia para isolar este perfil no mapa de Pareto acima.
                            </span>
                        </div>
                    </h3>
                    <div class="chart-explainer">
                        <div class="explainer-col citizen">
                            <h4 style="color:var(--citizen)">Gestão de Operação:</h4>
                            O robô separou as escolas. A fatia "C4: Super-Polos" abriga os monstros logísticos do país. Nunca envie coordenadores inexperientes para estes locais.
                        </div>
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Validação Científica:</h4>
                            Clusterização não-supervisionada particionando dados via minimização da Inércia intra-cluster após StandardScaler.
                        </div>
                    </div>
                    <div id="chartIA" class="plotly-chart"></div>
                </div>
                
                <!-- CHART 3: FAIXAS INTERATIVAS -->
                <div class="card">
                    <h3>
                        C. Estratificação de Quartis: A Variável FAIXA_CAPACIDADE
                        <div class="tooltip-icon">i
                            <span class="tooltip-text">
                                <strong>Categorização Tukey</strong><br>
                                <span class="tooltip-formula">Micro: <= Q1</span>
                                <span class="tooltip-formula">Pequena: > Q1 a <= Q2</span>
                                <span class="tooltip-formula">Média: > Q2 a <= Q3</span>
                                <span class="tooltip-formula">Auditório: Outlier</span>
                            </span>
                        </div>
                    </h3>
                    <div class="chart-explainer">
                        <div class="explainer-col citizen">
                            <h4 style="color:var(--citizen)">Gestão de Operação:</h4>
                            A altura destas barras mostra qual é o prédio "padrão" do estado que você filtrou. Se a barra "Micro Sala" explodir, prepare-se para gastar com contratações de mais fiscais.
                        </div>
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Validação Científica:</h4>
                            Histograma categórico ordinal para validação da transposição de uma variável contínua (Capacity) para uma variável discreta de negócio.
                        </div>
                    </div>
                    <div id="chartFaixas" class="plotly-chart"></div>
                </div>
                
                <!-- CHART 4: SCATTER BIVARIADO -->
                <div class="card">
                    <h3>
                        D. Dispersão Bivariada: Prédios Pulverizados vs. Prédios Massivos
                        <div class="tooltip-icon">i
                            <span class="tooltip-text">
                                <strong>Gráfico Scatter Plot</strong><br>
                                Cada ponto representa a agregação (GROUP BY) de um Local Físico.<br>
                                <span class="tooltip-formula">X: CONTAGEM DE SALAS</span>
                                <span class="tooltip-formula">Y: MÉDIA DA CAPACIDADE</span>
                            </span>
                        </div>
                    </h3>
                    <div class="chart-explainer">
                        <div class="explainer-col citizen">
                            <h4 style="color:var(--citizen)">Gestão de Operação:</h4>
                            Subir significa "salas gigantes". Direita significa "muitas salas em um só prédio". Bolinhas no canto superior direito são as escolas mais difíceis de se controlar no Brasil inteiro.
                        </div>
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Validação Científica:</h4>
                            Análise bidimensional isolando Qtd e Média. Outliers superiores direitos representam a máxima inércia estrutural do conjunto.
                        </div>
                    </div>
                    <div id="chartScatter" class="plotly-chart"></div>
                </div>

                <!-- CHART 5: TOP 20 BARRAS HORIZONTAIS -->
                <div class="card">
                    <h3>E. Polos Magnos: Top 20 Locais por Capacidade Concentrada</h3>
                    <div class="chart-explainer">
                        <div class="explainer-col citizen">
                            <h4 style="color:var(--citizen)">Visão do Gestor:</h4>
                            Ranking que muda de acordo com seus filtros laterais. Estes são os alicerces críticos da sua seleção.
                        </div>
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Validação Científica:</h4>
                            Recorte matricial (Head 20) com reordenação paramétrica. Facilita auditorias cirúrgicas rápidas.
                        </div>
                    </div>
                    <div id="chartTop20" class="plotly-chart"></div>
                </div>

                <!-- GRÁFICOS ESTÁTICOS MACRO -->
                <h2 id="estaticos" style="border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-top: 50px;">
                    3. MODELOS GLOBAIS E ESTÁTICOS 
                    <div class="tooltip-icon">i<span class="tooltip-text"><strong>Modelagem de Base Fixa</strong><br>Estes gráficos não são afetados pelos filtros em Javascript pois representam funções computacionais pesadas (ECDF, Pearson) executadas na compilação do Python. Eles atestam a sanidade do País como um todo.</span></div>
                </h2>
                
                <div class="card">
                    <h3>F. Avaliação de Assimetria e Probabilidade Contínua (ECDF)</h3>
                    <div class="chart-explainer">
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Parecer da Estatística:</h4>
                            O Histograma à direita atesta graficamente o grau de Skewness da base. A curva ECDF indica qual a probabilidade de uma sala escolhida ao acaso estar dentro de um determinado limite logístico.
                        </div>
                    </div>
                    <iframe src="graficos/2_Histograma_Distribuicao.html"></iframe>
                    <iframe src="graficos/13_ECDF_Acumulada.html"></iframe>
                </div>
                
                <div class="card">
                    <h3>G. Estudo de Dispersão e Outliers em Violino</h3>
                    <div class="chart-explainer">
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Parecer da Estatística:</h4>
                            O Boxplot revela que a categoria 'Gigante/Auditório' possui Amplitude Interquartílica severa (despadronizada). O Violino mostra a densidade populacional contínua (KDE) dentro de cada UF do país.
                        </div>
                    </div>
                    <iframe src="graficos/9_Boxplot_Faixas.html"></iframe>
                    <iframe src="graficos/4_Violino_Densidade.html"></iframe>
                </div>

                <div class="card">
                    <h3>H. Matriz de Correlação Linear e Waterfall (Cascata)</h3>
                    <div class="chart-explainer">
                        <div class="explainer-col tech">
                            <h4 style="color:var(--tech)">Parecer da Estatística:</h4>
                            O Heatmap atesta se a 'Quantidade de Salas' e a 'Capacidade' caminham perfeitamente juntas (Correlação > 0.8). O Waterfall prova matematicamente o peso da Cauda Longa ('Demais Estados') se comparada às 10 maiores UFs somadas.
                        </div>
                    </div>
                    <iframe src="graficos/12_Heatmap_Correlacao.html"></iframe>
                    <iframe src="graficos/10_Waterfall_UFs.html"></iframe>
                </div>
                
                <div class="card">
                    <h3>I. Mapa de Concentração Hierárquico (Treemap Global)</h3>
                    <iframe src="graficos/3_Treemap_Concentracao.html"></iframe>
                </div>

                <h2 id="tabela" style="border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-top: 50px;">4. DATA GRID EXPLORATÓRIO (VINCULADO AOS FILTROS)</h2>
                <div class="data-table-container">
                    <p style="margin-top:0;"><strong>Instrução:</strong> Esta tabela (DataTables.js) reflete cirurgicamente a matriz de dados baseada nos seus cliques. Utilize a caixa <strong>Search</strong> para realizar mineração textual de escolas ou municípios.</p>
                    <table id="mainTable" class="display" style="width:100%">
                        <thead>
                            <tr>
                                <th>UF</th>
                                <th>Localização / Escola</th>
                                <th>Nº Sala</th>
                                <th>Capacidade (Vagas)</th>
                                <th>Faixa Estrutural (Quartil)</th>
                                <th>Perfil IA (Risco KMeans)</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>

            </div>

            <!-- MOTOR JAVASCRIPT RECONSTRUÍDO (CROSS-FILTERING E ESTATÍSTICA) -->
            <script>
                const rawData = {json_str};
                let dataTable;
                const natTotSalas = rawData.length;
                
                function calcMean(arr) {{ return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }}
                function calcMedian(arr) {{
                    if(!arr.length) return 0;
                    let s = [...arr].sort((a,b)=>a-b);
                    let mid = Math.floor(s.length/2);
                    return s.length % 2 === 0 ? (s[mid-1]+s[mid])/2 : s[mid];
                }}
                function calcStdDev(arr, mean) {{
                    if(arr.length < 2) return 0;
                    let varSum = arr.reduce((a,b) => a + Math.pow(b - mean, 2), 0);
                    return Math.sqrt(varSum / (arr.length - 1));
                }}
                function calcSkewness(arr, mean, std) {{
                    if(arr.length < 3 || std === 0) return 0;
                    let n = arr.length;
                    let skewSum = arr.reduce((a,b) => a + Math.pow((b - mean)/std, 3), 0);
                    return (n / ((n-1)*(n-2))) * skewSum;
                }}

                $(document).ready(function() {{
                    dataTable = $('#mainTable').DataTable({{
                        data: rawData,
                        columns: [
                            {{ data: 'SG_UF' }},
                            {{ data: 'NO_LOCAL' }},
                            {{ data: 'NO_SALA' }},
                            {{ data: 'QT_CAPACIDADE_MAXIMA_SALA' }},
                            {{ data: 'FAIXA_CAPACIDADE' }},
                            {{ data: 'PERFIL_COMPORTAMENTAL_IA' }}
                        ],
                        pageLength: 15,
                        language: {{ url: '//cdn.datatables.net/plug-ins/1.13.6/i18n/pt-BR.json' }}
                    }});
                    populateDropdowns();
                    updateDashboard();
                }});

                function populateDropdowns() {{
                    const ufs = [...new Set(rawData.map(d => d.SG_UF))].filter(Boolean).sort();
                    const faixas = [...new Set(rawData.map(d => d.FAIXA_CAPACIDADE))].filter(Boolean).sort();
                    const clusters = [...new Set(rawData.map(d => d.PERFIL_COMPORTAMENTAL_IA))].filter(Boolean).sort();
                    
                    ufs.forEach(uf => $('#filterUF').append(new Option(uf, uf)));
                    faixas.forEach(fx => $('#filterFaixa').append(new Option(fx, fx)));
                    clusters.forEach(cl => $('#filterCluster').append(new Option(cl, cl)));
                }}

                function resetFilters() {{
                    $('#filterUF').val('ALL');
                    $('#filterFaixa').val('ALL');
                    $('#filterCluster').val('ALL');
                    updateDashboard();
                }}

                function updateDashboard() {{
                    const uf = $('#filterUF').val();
                    const faixa = $('#filterFaixa').val();
                    const cluster = $('#filterCluster').val();

                    const filteredData = rawData.filter(d => {{
                        return (uf === 'ALL' || d.SG_UF === uf) &&
                               (faixa === 'ALL' || d.FAIXA_CAPACIDADE === faixa) &&
                               (cluster === 'ALL' || d.PERFIL_COMPORTAMENTAL_IA === cluster);
                    }});

                    // 1. Atualizar Tabela (DataGrid)
                    dataTable.clear().rows.add(filteredData).draw();

                    // 2. Extração para Estatística Nativa no JS
                    const capacidades = filteredData.map(d => d.QT_CAPACIDADE_MAXIMA_SALA || 0);
                    const totCapacidade = capacidades.reduce((sum, val) => sum + val, 0);
                    const totSalas = filteredData.length;
                    const locaisUnicos = new Set(filteredData.map(d => d.NO_LOCAL)).size;
                    
                    const mean = calcMean(capacidades);
                    const median = calcMedian(capacidades);
                    const std = calcStdDev(capacidades, mean);
                    const skew = calcSkewness(capacidades, mean, std);

                    // 3. Renderizar KPIs Visuais
                    document.getElementById('kpiCapacidade').innerText = totCapacidade.toLocaleString('pt-BR');
                    document.getElementById('kpiSalas').innerText = totSalas.toLocaleString('pt-BR');
                    document.getElementById('kpiLocais').innerText = locaisUnicos.toLocaleString('pt-BR');
                    document.getElementById('kpiMedia').innerText = mean.toFixed(1);

                    // 4. Data Storytelling e Perfil Parcial
                    let isNacional = (totSalas === natTotSalas);
                    let perfilTexto = "";
                    if (isNacional) {{
                        perfilTexto = "Você está visualizando a Base Nacional Bruta. Os cálculos representam a totalidade física e logística mapeada na tabela original.";
                    }} else {{
                        let labelFiltro = [];
                        if (uf !== 'ALL') labelFiltro.push(`UF: ${{uf}}`);
                        if (faixa !== 'ALL') labelFiltro.push(`Faixa: ${{faixa}}`);
                        if (cluster !== 'ALL') labelFiltro.push(`Perfil: ${{cluster}}`);
                        perfilTexto = `Você isolou a análise para [ <strong>${{labelFiltro.join(' | ')}}</strong> ]. Este cruzamento exibe comportamento logístico unicamente desta seleção (Cross-Filtering Ativo).`;
                    }}
                    
                    document.getElementById('perfil-narrativa').innerHTML = perfilTexto;
                    document.getElementById('pa-rep').innerText = ((totSalas / natTotSalas) * 100).toFixed(1) + "%";
                    document.getElementById('pa-med').innerText = median.toFixed(1) + " lugs";
                    document.getElementById('pa-std').innerText = std.toFixed(1);
                    document.getElementById('pa-skew').innerText = skew.toFixed(2);

                    // 5. Alertas e Inteligência (Insight Box)
                    const insightBox = document.getElementById('insight-box');
                    const insightText = document.getElementById('insight-text');
                    insightBox.style.display = 'none';
                    
                    if (!isNacional && totSalas > 0) {{
                        if (skew > 1.5) {{
                            insightBox.style.display = 'block'; insightBox.style.borderLeftColor = '#e74c3c';
                            insightText.innerHTML = `<strong>Alerta de Assimetria e Falso Positivo:</strong> A assimetria deste grupo é severa (${{skew.toFixed(2)}}). Embora você tenha feito um recorte em "${{uf}}", existem raros prédios gigantescos aqui dentro puxando as médias estatísticas para o teto ilusoriamente. A Média é Mentirosa aqui.`;
                        }} else if (mean > 0 && mean < 25) {{
                            insightBox.style.display = 'block'; insightBox.style.borderLeftColor = '#f39c12';
                            insightText.innerHTML = `<strong>Risco Logístico de Capilaridade:</strong> A capacidade média deste recorte (${{mean.toFixed(1)}}) é extremamente baixa. A logística e coordenação de equipes aqui serão custosas, pois os candidatos estão espalhados em milhares de salinhas apertadas.`;
                        }}
                    }}

                    // ============================================
                    // PLOTLY JAVASCRIPT ENGINE (O CORAÇÃO DO CROSS-FILTERING)
                    // ============================================

                    // A. PLOTLY PARETO REATIVO
                    const ufSums = {{}};
                    filteredData.forEach(d => {{ if(d.SG_UF) ufSums[d.SG_UF] = (ufSums[d.SG_UF] || 0) + (d.QT_CAPACIDADE_MAXIMA_SALA || 0); }});
                    const sortedUFs = Object.keys(ufSums).sort((a,b) => ufSums[b] - ufSums[a]);
                    const paretoY = sortedUFs.map(k => ufSums[k]);
                    let acc = 0; const paretoPerc = paretoY.map(v => {{ acc += v; return (acc/totCapacidade)*100; }});
                    
                    Plotly.react('chartPareto', [
                        {{ x: sortedUFs, y: paretoY, type: 'bar', name: 'Volumetria Bruta', marker: {{color: '#0f4c81'}} }},
                        {{ x: sortedUFs, y: paretoPerc, type: 'scatter', name: 'Curva Cumulativa (%)', yaxis: 'y2', line: {{color: '#e74c3c', width: 3}} }}
                    ], {{ yaxis2: {{overlaying: 'y', side: 'right', range: [0, 105]}}, legend: {{x: 0, y: 1.1, orientation: 'h'}}, margin: {{t:20}} }});
                    
                    document.getElementById('chartPareto').removeAllListeners('plotly_click');
                    document.getElementById('chartPareto').on('plotly_click', function(data){{
                        if(data.points[0].x) {{ $('#filterUF').val(data.points[0].x); updateDashboard(); }}
                    }});

                    // B. PLOTLY PIZZA DE IA (CLUSTERS)
                    const iaCounts = {{}};
                    filteredData.forEach(d => {{ if(d.PERFIL_COMPORTAMENTAL_IA) iaCounts[d.PERFIL_COMPORTAMENTAL_IA] = (iaCounts[d.PERFIL_COMPORTAMENTAL_IA] || 0) + 1; }});
                    Plotly.react('chartIA', [{{ values: Object.values(iaCounts), labels: Object.keys(iaCounts), type: 'pie', hole: 0.4 }}], {{ margin: {{t:20}} }});
                    
                    document.getElementById('chartIA').removeAllListeners('plotly_click');
                    document.getElementById('chartIA').on('plotly_click', function(data){{
                        if(data.points[0].label) {{ $('#filterCluster').val(data.points[0].label); updateDashboard(); }}
                    }});

                    // C. PLOTLY BARRAS DE FAIXA DE CAPACIDADE
                    const faixaCounts = {{}};
                    filteredData.forEach(d => {{ if(d.FAIXA_CAPACIDADE) faixaCounts[d.FAIXA_CAPACIDADE] = (faixaCounts[d.FAIXA_CAPACIDADE] || 0) + 1; }});
                    Plotly.react('chartFaixas', [{{ x: Object.keys(faixaCounts).sort(), y: Object.keys(faixaCounts).sort().map(k => faixaCounts[k]), type: 'bar', marker: {{color: '#2980b9'}} }}], {{ margin: {{t:20}} }});
                    
                    document.getElementById('chartFaixas').removeAllListeners('plotly_click');
                    document.getElementById('chartFaixas').on('plotly_click', function(data){{
                        if(data.points[0].x) {{ $('#filterFaixa').val(data.points[0].x); updateDashboard(); }}
                    }});

                    // D. PLOTLY SCATTER PLOT (DISPERSÃO DE PRÉDIOS)
                    const localAgg = {{}};
                    filteredData.forEach(d => {{
                        if(!localAgg[d.NO_LOCAL]) localAgg[d.NO_LOCAL] = {{s: 0, c: 0, p: d.PERFIL_COMPORTAMENTAL_IA}};
                        localAgg[d.NO_LOCAL].s += 1; localAgg[d.NO_LOCAL].c += d.QT_CAPACIDADE_MAXIMA_SALA || 0;
                    }});
                    const scatterX=[], scatterY=[], scatterText=[], scatterColor=[];
                    for(let loc in localAgg) {{
                        scatterText.push(loc); scatterX.push(localAgg[loc].s);
                        scatterY.push(localAgg[loc].c / localAgg[loc].s); scatterColor.push(localAgg[loc].p);
                    }}
                    const cmap = {{'C1': '#3498db', 'C2': '#f1c40f', 'C3': '#e67e22', 'C4': '#e74c3c'}};
                    const pointColors = scatterColor.map(c => {{
                        for(let key in cmap) if(c && c.includes(key)) return cmap[key];
                        return '#95a5a6';
                    }});
                    Plotly.react('chartScatter', [{{ x: scatterX, y: scatterY, text: scatterText, mode: 'markers', marker: {{size: 8, color: pointColors, opacity: 0.7}} }}], {{ xaxis: {{title: 'Total de Salas Agrupadas no Prédio'}}, yaxis: {{title: 'Capacidade Média Logística (Cadeiras)'}} }});
                    
                    // E. PLOTLY TOP 20 BARRAS HORIZONTAIS
                    const localSumArr = Object.entries(localAgg).map(([k,v]) => [k, v.s * (v.c/v.s)]).sort((a,b)=>b[1]-a[1]).slice(0,20).reverse();
                    Plotly.react('chartTop20', [{{ y: localSumArr.map(d=>d[0]), x: localSumArr.map(d=>d[1]), type: 'bar', orientation: 'h', marker: {{color: '#8e44ad'}} }}], {{ margin: {{t:20, l:150}} }});
                }}
            </script>
        </body>
        </html>
        """
        with open(f"{self.output_dir}/2_Dashboard_Executivo_Didatico.html", "w", encoding="utf-8") as f:
            f.write(html_content)

    def export_didactic_report(self, geral_stats, uf_stats, faixas_df, cutoffs):
        logger.info("Elaborando Laudo Analítico Técnico/Cidadão consolidado (TXT)...")
        texto = f"""=============================================================================
LAUDO TÉCNICO E CIDADÃO: DIAGNÓSTICO PROFUNDO DE INFRAESTRUTURA LOGÍSTICA
(Documento Unificado - Padrão Consultoria Estratégica BI)
=============================================================================

PARTE 1: VISÃO EXECUTIVA E CIDADÃ (O RESUMO SIMPLIFICADO)
-----------------------------------------------------------------------------
O QUE FIZEMOS: Um algoritmo varreu todas as salas do país, corrigiu 
erros de digitação antigos (SAS/Excel) e calculou a capacidade real do Brasil. 
As "Salas Médias" (entre {cutoffs.get('q50', 0)+1} e {cutoffs.get('q75', 0)} lugares) são a prova de que temos um 
padrão seguro de trabalho. Já os locais classificados pela Inteligência Artificial 
como "C4: Super-Polos" e as salas "Gigantes/Auditórios" são as zonas vermelhas 
do projeto. Neles, falhar não é uma opção; coloque sua melhor equipe lá.

PARTE 2: LAUDO TÉCNICO-CIENTÍFICO (PROFUNDIDADE ESTATÍSTICA)
-----------------------------------------------------------------------------
1. ARQUITETURA DE DADOS E TRANSPARÊNCIA:
O Data Pipeline desabilitou injeções de artefatos hexadecimais (Excel vazado) 
e executou um parsing regressivo do charset cp1252 (SAS/Windows) para UTF-8, 
garantindo preservação integral da integridade referencial dos nomes de escolas.

2. INFERÊNCIA E DISPERSÃO ESTATÍSTICA:
Constata-se uma Assimetria fortemente enviesada ({geral_stats.loc['Assimetria (Skewness)', 'Valor Estimado']:.2f}). A validação visual via 
Boxplot (disponível no relatório HTML estático) corrobora a tese de que a amplitude 
interquartílica (IQR) é ínfima se comparada à extensão da cauda longa logística.
É tecnicamente imperativo que as métricas de Tomada de Decisão substituam a 
utilização empírica da Média Paramétrica ({geral_stats.loc['Média Aritmética (\u03BC)', 'Valor Estimado']:.1f}) pela Mediana ({geral_stats.loc['Mediana (Ponto de Separação)', 'Valor Estimado']:.1f}).

3. MACHINE LEARNING E CONCENTRAÇÃO:
Os K-Means isolaram a variabilidade estrutural. O cruzamento das bases atesta 
que focar esforços de auditoria exclusivamente nas UFs que compõem os 80% do 
teorema de Pareto blindará sistemicamente a execução operacional do projeto.
"""
        with open(f"{self.output_dir}/3_Laudo_Insights_Consolidados.txt", "w", encoding="utf-8") as f:
            f.write(texto)

    def export_excel_corporate(self, df_dict, filename="4_Planilha_Consultoria_Master.xlsx", cutoffs=None):
        logger.info("Gerando Master Spreadsheet com Formatação Condicional (Data Bars/Color Scales)...")
        filepath = f"{self.output_dir}/{filename}"
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        workbook = writer.book

        title_format = workbook.add_format({'bold': True, 'font_size': 13, 'font_color': '#1F4E78', 'bg_color': '#D9E1F2', 'border': 1, 'align': 'left', 'valign': 'vcenter'})
        desc_format = workbook.add_format({'text_wrap': True, 'italic': True, 'font_size': 10, 'font_color': '#333333', 'bg_color': '#F2F2F2', 'border': 1, 'valign': 'top'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#1F4E78', 'font_color': 'white', 'border': 1})
        cell_format = workbook.add_format({'border': 1})

        explica_abas = {
            '1_Base_Tratada': ('BASE DE DADOS HIGIENIZADA (DATA PIPELINE)', 'VISÃO LEIGA: Nomes limpos e fáceis de ler. | VISÃO TÉCNICA: Artefatos Hexadecimais e Mojibakes do SAS revertidos algoritmicamente.'),
            '2_Estat_Avançadas': ('INFERÊNCIA ESTATÍSTICA E ESTUDO DE DISTRIBUIÇÃO', 'VISÃO LEIGA: A "Mediana" é o meio exato; o "Desvio" mede a desordem. | VISÃO TÉCNICA: Base leptocúrtica com forte assimetria, exigindo cautela no uso de média simples.'),
            '3_Analise_UF': ('DISPERSÃO GEOGRÁFICA E PESO ESTATAL', 'VISÃO LEIGA: Qual estado importa mais no projeto. | VISÃO TÉCNICA: Matriz de densidade demográfica predial para rateio orçamentário. (Barras de Dados aplicadas).'),
            '4_Top_20_Locais': ('PONTOS CRÍTICOS: TOP 20 SUPER-POLOS', 'VISÃO LEIGA: As 20 escolas que não podem falhar. | VISÃO TÉCNICA: Instalações C4 onde a matriz de risco impõe gestão direta. (Data Scale aplicado).'),
            '5_Pareto_80_20': ('TEOREMA DE PARETO (EFICIÊNCIA DE ALOCAÇÃO)', 'VISÃO LEIGA: Resolva o topo e garanta 80% da paz nacional. | VISÃO TÉCNICA: Curva logarítmica de mitigação de risco operacional.'),
            '6_Clusters_IA': ('SEGMENTAÇÃO DA INTELIGÊNCIA ARTIFICIAL (PERFIL)', 'VISÃO LEIGA: A IA diz como tratar o prédio (C1 a C4). | VISÃO TÉCNICA: K-Means agrupando locais para alocação sênior/júnior baseada em similaridade.'),
            '7_Outliers_Anomalias': ('DETECÇÃO DE ANOMALIAS (OUTLIERS / ERROS)', 'VISÃO LEIGA: Escolas gigantescas ou com "erro de dedo" (zeros a mais). | VISÃO TÉCNICA: Z-Scores absolutos excedendo limiares padrões.')
        }

        ws_guia = workbook.add_worksheet('0_GUIA_E_GLOSSARIO')
        ws_guia.set_column('A:A', 130)
        ws_guia.write('A1', 'DICIONÁRIO UNIVERSAL E GUIA CIENTÍFICO (MANUAL DE LEITURA)', title_format)
        
        texto_cutoffs = ""
        if cutoffs:
            texto_cutoffs = (
                r"\n\nO QUE É A VARIÁVEL [FAIXA_CAPACIDADE]? (Regra dos Quartis Matemáticos)\n"
                f"► 1. Micro Sala (Até {cutoffs.get('q25', 0)} lugares): Limite do Quartil 1. Exige fragmentar muitas pessoas em várias salinhas (Difícil controle logístico).\n"
                f"► 2. Sala Pequena ({cutoffs.get('q25', 0)+1} a {cutoffs.get('q50', 0)} lugares): Volume padrão mediano.\n"
                f"► 3. Sala Média ({cutoffs.get('q50', 0)+1} a {cutoffs.get('q75', 0)} lugares): Eixo de Q3. Trade-off perfeito de eficiência e controle.\n"
                f"► 4. Sala Grande ({cutoffs.get('q75', 0)+1} a {cutoffs.get('limite_sup', 0)} lugares): Extremidade da Curva Normal. Locais massificados.\n"
                f"► 5. Auditório/Gigante (Acima de {cutoffs.get('limite_sup', 0)} lugares): São os Outliers. Instalações fora do padrão normal. Precisam ser fiscalizadas uma a uma."
            ).replace(r"\n", "\n")

        guia_texto = (
            "Este arquivo é um Data Warehouse consolidado por Python, mesclando Linguagem Cidadã e Científica.\n\n"
            "DICA DE OURO PARA GESTORES:\n"
            "► TODA ABA tem um cabeçalho cinza traduzindo os números complexos para português claro e indicando a decisão técnica por trás dele.\n"
            "► COMECE revisando a aba '7_Outliers_Anomalias'. Ela limpa a sujeira crítica da sua operação."
            + texto_cutoffs
        )
        ws_guia.merge_range('A2:A20', guia_texto, desc_format)

        for sheet_name, df_sheet in df_dict.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, startrow=4, index=True if sheet_name == '2_Estat_Avançadas' else False)
            worksheet = writer.sheets[sheet_name]
            
            titulo, explicacao = explica_abas.get(sheet_name, ('', ''))
            col_max = max(len(df_sheet.columns) - 1, 5)
            if sheet_name == '2_Estat_Avançadas': col_max = 5
            
            worksheet.merge_range(0, 0, 0, col_max, titulo, title_format)
            worksheet.merge_range(1, 0, 2, col_max, explicacao, desc_format)
            worksheet.freeze_panes(5, 0)
            
            if sheet_name != '2_Estat_Avançadas':
                worksheet.autofilter(4, 0, len(df_sheet) + 4, len(df_sheet.columns) - 1)
            
            for col_num, value in enumerate(df_sheet.columns):
                idx_col = col_num if sheet_name != '2_Estat_Avançadas' else col_num+1
                worksheet.write(4, idx_col, str(value), header_format)
                worksheet.set_column(idx_col, idx_col, 25, cell_format)
                
            if sheet_name == '2_Estat_Avançadas':
                worksheet.write(4, 0, 'MÉTRICA ESTATÍSTICA (CONSOLIDADA)', header_format)
                worksheet.set_column(0, 0, 50, cell_format)
                
            # Formatação Condicional Inteligente
            if sheet_name == '3_Analise_UF':
                worksheet.conditional_format(5, 4, len(df_sheet)+4, 4, {'type': 'data_bar', 'bar_color': '#5A8AC6'})
            elif sheet_name == '4_Top_20_Locais':
                worksheet.conditional_format(5, 4, len(df_sheet)+4, 4, {'type': '3_color_scale', 'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63C384'})
            elif sheet_name == '5_Pareto_80_20':
                worksheet.conditional_format(5, 3, len(df_sheet)+4, 3, {'type': 'data_bar', 'bar_color': '#5A8AC6'})

        writer.close()


class SystemOrchestrator:
    """Orquestrador mantido com adição de suporte ao Streamlit para callbacks visuais."""
    def __init__(self, filepath, output_dir="Output_Analise_Salas"):
        self.filepath = filepath
        self.output_dir = output_dir
        self.results = {} # Container de memória para uso posterior no Streamlit

    def run(self, progress_bar=None, status_text=None):
        logger.info("🚀 Inicializando Plataforma Master BI (Versão 7.1 - UI Integrada com Fix Mismatch)...")
        
        def update_ui(msg, val):
            if status_text: status_text.text(msg)
            if progress_bar: progress_bar.progress(val)
            print(msg)
            
        try:
            update_ui("Etapa 1/7: Pipeline de Limpeza e Sanitização...", 0.1)
            loader = DataLoaderAndCleaner(self.filepath, fix_encoding=True)
            loader.load_data()
            df_clean, data_quality = loader.clean_and_validate()
            self.results['data_quality'] = data_quality
            
            update_ui("Etapa 2/7: Computando Inteligência Estatística...", 0.25)
            analyzer = StatisticalAnalyzer(df_clean)
            faixas_df, cutoffs = analyzer.apply_capacity_bins()
            geral_stats = analyzer.get_general_stats()
            uf_stats = analyzer.group_analysis(['SG_UF']) if 'SG_UF' in df_clean.columns else pd.DataFrame()
            outliers = analyzer.detect_outliers()
            pareto_uf = analyzer.pareto_analysis('SG_UF') if 'SG_UF' in df_clean.columns else pd.DataFrame()
            top_locais = analyzer.get_top_locais(20)
            
            self.results['df_clean'] = df_clean
            self.results['geral_stats'] = geral_stats
            self.results['outliers'] = outliers
            
            update_ui("Etapa 3/7: Treinando Machine Learning (K-Means)...", 0.4)
            ml_engine = MachineLearningEngine(df_clean)
            clusters = ml_engine.cluster_locations()
            df_clean = ml_engine.df 
            
            visualizer = VisualizerAndExporter(self.output_dir)
            
            update_ui("Etapa 4/7: Salvando Metadados JSON e CSV...", 0.55)
            insights_json = {"qualidade": data_quality, "outliers": len(outliers), "cutoffs": cutoffs}
            with open(f"{self.output_dir}/5_Metadados_Insights.json", "w", encoding="utf-8") as f:
                json.dump(insights_json, f, indent=4, ensure_ascii=False)
            df_clean.to_csv(f"{self.output_dir}/6_Base_Tratada.csv", index=False, encoding='utf-8-sig', sep=';')
            
            update_ui("Etapa 5/7: Compilando 13 Painéis e Single Page App...", 0.7)
            visualizer.generate_charts(df_clean, uf_stats, pareto_uf, top_locais, faixas_df, clusters)
            visualizer.generate_html_report(df_clean)
            visualizer.create_custom_portuguese_dashboard(df_clean, cutoffs, geral_stats)
            self.results['figs'] = visualizer.generated_figs # Passa as figuras para UI nativa
            
            update_ui("Etapa 6/7: Redigindo Laudo Textual...", 0.85)
            visualizer.export_didactic_report(geral_stats, uf_stats, faixas_df, cutoffs)
            
            update_ui("Etapa 7/7: Construindo Master Spreadsheet Excel...", 0.95)
            tabelas_excel = {
                '1_Base_Tratada': df_clean,
                '2_Estat_Avançadas': geral_stats,
                '3_Analise_UF': uf_stats,
                '4_Top_20_Locais': top_locais,
                '5_Pareto_80_20': pareto_uf,
                '6_Clusters_IA': clusters,
                '7_Outliers_Anomalias': outliers
            }
            visualizer.export_excel_corporate(tabelas_excel, cutoffs=cutoffs)
            
            update_ui("Processamento 100% Finalizado!", 1.0)
            logger.info("✅ Plataforma de BI embarcada concluída com sucesso.")
            return True
            
        except Exception as e:
            logger.error(f"Erro Crítico no Orquestrador: {e}")
            logger.error(traceback.format_exc())
            raise


# ==========================================
# CAMADA DE INTERFACE (STREAMLIT APP)
# ==========================================

def zip_directory(folder_path, zip_path):
    """Utilitário para empacotar toda a saída para download do usuário."""
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)

def main_streamlit():
    # Cabeçalho Moderno
    st.title("🚀 Plataforma Master BI: Analítica de Salas")
    st.markdown("""
    Bem-vindo ao **Pipeline Analítico**. Submeta sua base de dados bruta para desencadear um processo ponta-a-ponta de higienização automatizada, 
    estatística paramétrica avançada, modelagem de IA não supervisionada (K-Means) e geração de Painéis Interativos e Relatórios Executivos.
    """)
    st.divider()

    # Configuração de Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📥 1. Ingestão de Dados")
        uploaded_file = st.file_uploader("Faça upload da planilha original (.xlsx)", type=["xlsx"])
        
        # Área de log visual
        log_expander = st.expander("Terminal de Execução (Logs)", expanded=False)

    if uploaded_file is not None:
        with col2:
            st.subheader("⚙️ 2. Validação Estrutural")
            
            # Validação Leve Inicial sem travar a engine original
            try:
                df_preview = pd.read_excel(uploaded_file, nrows=5)
                cols_encontradas = list(df_preview.columns)
                colunas_obrigatorias = ['QT_CAPACIDADE_MAXIMA_SALA'] # Core indispensable
                colunas_recomendadas = ['NO_LOCAL', 'SG_UF']
                
                faltam_obrigatórias = [c for c in colunas_obrigatorias if c not in cols_encontradas]
                faltam_recomendadas = [c for c in colunas_recomendadas if c not in cols_encontradas]
                
                if faltam_obrigatórias:
                    st.error(f"🚨 ERRO ESTRUTURAL: A planilha não contém a(s) coluna(s) obrigatória(s): {', '.join(faltam_obrigatórias)}")
                    st.stop()
                
                st.success("✅ Arquivo validado! A estrutura básica está correta.")
                if faltam_recomendadas:
                    st.warning(f"⚠️ Aviso: Faltam colunas recomendadas para algumas análises ({', '.join(faltam_recomendadas)}). A pipeline executará em modo degradação suave.")
                
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo. Ele pode estar corrompido: {e}")
                st.stop()

            # Botão de ignição
            if st.button("▶️ INICIAR PROCESSAMENTO ANALÍTICO COMPLETO", type="primary", use_container_width=True):
                
                # Setup do Diretório Temporário para a execução
                temp_dir = tempfile.mkdtemp()
                output_folder = os.path.join(temp_dir, "Output_Analise_Salas")
                temp_input_file = os.path.join(temp_dir, "input_data.xlsx")
                
                # Salvar o arquivo upado para o script nativo rodar exatamente como rodava antes
                with open(temp_input_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.markdown("### Progresso da Execução")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                orchestrator = SystemOrchestrator(temp_input_file, output_folder)
                
                try:
                    with st.spinner("Processando arquitetura de dados e IA..."):
                        orchestrator.run(progress_bar, status_text)
                    
                    st.success("🎉 Processamento Finalizado com Sucesso!")
                    
                    # Atualiza os Logs no painel expansível
                    with log_expander:
                        st.code(log_stream.getvalue(), language='bash')
                    
                    # ---------------------------------------------
                    # APRESENTAÇÃO DE RESULTADOS NA PRÓPRIA UI
                    # ---------------------------------------------
                    st.divider()
                    st.subheader("📊 3. Resumo da Execução")
                    
                    # Métricas Extraídas
                    quality = orchestrator.results.get('data_quality', {})
                    df_clean = orchestrator.results.get('df_clean', pd.DataFrame())
                    stats_df = orchestrator.results.get('geral_stats', pd.DataFrame())
                    outliers_df = orchestrator.results.get('outliers', pd.DataFrame())
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Linhas Iniciais", quality.get('total_linhas_iniciais', 0))
                    m2.metric("Registros Válidos (Finais)", quality.get('total_linhas_finais', 0))
                    m3.metric("Anomalias/Outliers Detectados", len(outliers_df), delta_color="inverse")
                    m4.metric("Duplicadas Removidas", quality.get('duplicadas_encontradas', 0), delta_color="inverse")
                    
                    st.markdown("#### Pré-visualização do Relatório Estatístico")
                    if not stats_df.empty:
                        st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)
                    
                    # Exibir alguns Gráficos nativos do Streamlit como bônus (Sem perder o HTML original)
                    figs = orchestrator.results.get('figs', {})
                    if figs:
                        with st.expander("📈 Visualizar Gráficos Principais (Plotly Engine)"):
                            if '1_KPIs_Executivos' in figs: st.plotly_chart(figs['1_KPIs_Executivos'], use_container_width=True)
                            c1, c2 = st.columns(2)
                            with c1:
                                if '2_Histograma_Distribuicao' in figs: st.plotly_chart(figs['2_Histograma_Distribuicao'], use_container_width=True)
                                if '8_Clusters_IA_Distribuicao' in figs: st.plotly_chart(figs['8_Clusters_IA_Distribuicao'], use_container_width=True)
                            with c2:
                                if '9_Boxplot_Faixas' in figs: st.plotly_chart(figs['9_Boxplot_Faixas'], use_container_width=True)
                                if '5_Pareto_UF' in figs: st.plotly_chart(figs['5_Pareto_UF'], use_container_width=True)
                    
                    # ---------------------------------------------
                    # PACOTE DE DOWNLOAD CUMULATIVO
                    # ---------------------------------------------
                    st.divider()
                    st.subheader("📦 4. Exportação do Arsenal de BI")
                    st.info("Todo o pacote (Excel corporativo, Aplicação SPA Interativa em HTML, Relatório Sweetviz, Laudo em TXT, e Gráficos PGN/SVG/HTML) foi empacotado abaixo.")
                    
                    zip_path = os.path.join(temp_dir, "Plataforma_BI_Saida.zip")
                    zip_directory(output_folder, zip_path)
                    
                    with open(zip_path, "rb") as zfile:
                        st.download_button(
                            label="⬇️ BAIXAR TODOS OS RESULTADOS (.ZIP)",
                            data=zfile,
                            file_name="Plataforma_BI_Analitica.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error("🚨 Erro durante o processamento!")
                    st.error(str(e))
                    with log_expander:
                        st.code(traceback.format_exc(), language='python')
                    
                finally:
                    pass

if __name__ == "__main__":
    main_streamlit()
