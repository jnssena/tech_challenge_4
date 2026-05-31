# Tech Challenge Fase 4 - Pós Tech Data Analytics - FIAP
## Objetivo do Projeto
Desenvolver um modelo de Machine Learning para auxiliar um hospital a prever o nível de obesidade de um paciente com base em seus hábitos de vida e características físicas. O sistema foi deployado via Streamlit como uma aplicação preditiva hospitalar, contendo tanto um módulo de diagnóstico individual quanto um painel analítico com os principais insights sobre os dados.

## Ferramentas
- Python
- Jupyter Notebook
- Pandas e NumPy
- Matplotlib e Seaborn
- Scikit-learn
- Streamlit

## Metodologia
### 1. Coleta dos Dados:
Os dados foram obtidos a partir do dataset Obesity.csv, disponível no próprio repositório do GitHub, contendo características físicas, hábitos alimentares e estilo de vida de pacientes. O dicionário dos dados está contido no arquivo dicionario_obesity_fiap_tc4.pdf

### 2. Análise Exploratória:
Foram realizadas análises de distribuição da variável alvo, estatísticas descritivas das variáveis numéricas e geração de um gráfico de correlação entre as features. As variáveis categóricas foram codificadas via Label Encoding, com encoders individuais por coluna para permitir a inversão da transformação durante a predição. A variável alvo Obesity recebeu um encoder separado.

O Gradient Boosting foi selecionado como modelo final, atingindo 96,6% de assertividade.

### 3. Pipeline de Machine Learning:
O modelo final foi organizado em um `Pipeline` composto por duas etapas:
1. **StandardScaler** — normalização das features
2. **GradientBoostingClassifier** — classificação em 7 classes de obesidade

### 4. Deploy via Streamlit:
A aplicação foi deployada no Streamlit com duas abas principais:

- **Diagnóstico**: formulário lateral com os dados do paciente (características físicas, hábitos alimentares e estilo de vida), exibindo a classe prevista, confiança do modelo, IMC calculado e distribuição de probabilidade por classe.
- **Painel Analítico**: dashboard com KPIs e 4 gráficos — distribuição por gênero, boxplot de peso por classe, consumo de vegetais por nível de obesidade e frequência de atividade física por classe.

## Links
- **Streamlit**: https://techchallenge4-xeprkmuxye5didewydacxk.streamlit.app/
- **Apresentação do Projeto**: https://youtu.be/lPf8kOWVhRU