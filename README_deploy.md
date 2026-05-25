# 🚀 Guia de Deploy — ObesityAI no Streamlit Cloud

## O que você vai fazer (visão geral)
Você vai colocar o arquivo `app.py` no GitHub e conectar ao Streamlit Cloud.
O Streamlit Cloud vai rodar o app automaticamente e gerar um link público.

---

## Passo 1 — Adicionar os arquivos ao repositório do GitHub

Você precisa ter **dois arquivos** na raiz do seu repositório:
- `app.py` — o código do aplicativo
- `requirements.txt` — a lista de bibliotecas que o Streamlit precisa instalar

**Como fazer:**
1. Abra o repositório no GitHub: https://github.com/jnssena/tech_challenge_4
2. Clique em **"Add file" → "Upload files"**
3. Arraste os dois arquivos (`app.py` e `requirements.txt`) para a área de upload
4. Clique em **"Commit changes"**

---

## Passo 2 — Criar conta no Streamlit Cloud

1. Acesse: https://share.streamlit.io
2. Clique em **"Sign up"** e entre com sua conta do **GitHub**
   (use a mesma conta que tem o repositório)
3. Autorize o Streamlit a acessar seus repositórios

---

## Passo 3 — Fazer o Deploy

1. Já logado no Streamlit Cloud, clique em **"New app"**
2. Preencha as informações:
   - **Repository:** `jnssena/tech_challenge_4`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Clique em **"Deploy!"**
4. Aguarde 2–3 minutos (o Streamlit instala as bibliotecas automaticamente)
5. Seu app vai abrir com um link público no formato:
   `https://jnssena-tech-challenge-4-app-XXXXX.streamlit.app`

---

## O que é "deploy"?

Deploy significa **colocar sua aplicação em um servidor na internet**
para que outras pessoas possam acessar sem precisar rodar nada no computador.

Antes do deploy:
- O app roda só no SEU computador
- Ninguém mais consegue ver

Depois do deploy:
- O app tem uma URL pública
- Qualquer pessoa com o link consegue usar
- Funciona 24h, mesmo com seu computador desligado

---

## Sobre o joblib (o que você mencionou na aula)

O `joblib` é usado para **salvar o modelo treinado em um arquivo** (.pkl ou .joblib).
Sem ele, o modelo precisa ser treinado do zero toda vez que o app inicia.

Neste app, optamos por treinar o modelo no próprio app usando `@st.cache_resource`,
que é a forma recomendada no Streamlit — ele treina uma vez e guarda na memória
enquanto o servidor estiver rodando. Mais simples para o seu caso!

Se quiser usar joblib no futuro:
```python
import joblib
# Para salvar:
joblib.dump(modelo_final, 'modelo.pkl')
# Para carregar:
modelo = joblib.load('modelo.pkl')
```

---

## Estrutura recomendada do repositório

```
tech_challenge_4/
├── app.py                          ← código do Streamlit
├── requirements.txt                ← bibliotecas necessárias
├── modelo_obesity.ipynb            ← seu notebook de desenvolvimento
└── Referencias_Atividade/
    └── Obesity.csv
```

