import streamlit as st
import os
import requests
import base64
from PIL import Image
import io

# Configuração da página do Streamlit
st.set_page_config(page_title="Analisador Clínico Avançado", page_icon="🩺", layout="centered")

st.title("🩺 Analisador Clínico e Auditor de Relatórios Médicos (HF)")
st.write("Faça o upload do relatório para uma análise detalhada baseada em evidências.")

# Pega o Token do Hugging Face das variáveis de ambiente
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.info("Por favor, adicione seu HF_TOKEN nas variáveis de ambiente para continuar.", icon="🔑")
else:
    # Componente de upload de arquivo
    uploaded_file = st.file_uploader("Escolha o arquivo do relatório (JPG ou PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.success("Arquivo carregado com sucesso!")
        
        # Exibe uma prévia da imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Prévia do documento enviado", use_column_width=True)

        if st.button("🧬 Iniciar Análise Documental Avançada"):
            with st.spinner("O modelo do Hugging Face está processando o documento..."):
                try:
                    # Converte a imagem para Base64 para enviar via API
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Definição do modelo multimodal da Meta no Hugging Face
                    API_URL = "https://api.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    
                    # Seu prompt altamente estruturado (o mesmo que você definiu)
                    prompt_texto = """
                    Você é um médico especialista em análise documental clínica, com amplo conhecimento em medicina baseada em evidências e classificação internacional de doenças (CID-10 e CID-11).
                    Sua tarefa é analisar cuidadosamente o relatório médico enviado na imagem.

                    Responda estruturando sua resposta rigorosamente com os tópicos pedidos anteriormente (Etapa 1 à Etapa 11), usando Markdown para formatação e tabelas.
                    """

                    # Payload formatado para modelos de visão (Chat Completion)
                    payload = {
                        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_texto},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                                ]
                            }
                        ],
                        "max_tokens": 2000
                    }
                    
                    # Chamada HTTP para a API do Hugging Face
                    response = requests.post(API_URL, headers=headers, json=payload)
                    result = response.json()
                    
                    # Exibe o resultado na tela
                    st.markdown("---")
                    st.subheader("📑 Relatório de Análise Médica Gerado")
                    
                    # Trata a resposta que costuma vir na estrutura de chat
                    if "choices" in result:
                        texto_resposta = result["choices"][0]["message"]["content"]
                        st.markdown(texto_resposta)
                    elif "generated_text" in result:
                        st.markdown(result["generated_text"])
                    else:
                        st.markdown(result)
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar a análise clínica: {e}")
