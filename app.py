import streamlit as st
import os
import io
import base64
from PIL import Image
import requests

# Configuração da página do Streamlit
st.set_page_config(page_title="Analisador Clínico Avançado", page_icon="🩺", layout="centered")

st.title("🩺 Analisador Clínico e Auditor de Relatórios Médicos (HF)")
st.write("Faça o upload do relatório para uma análise detalhada baseada em evidências.")

# Pega o Token do Hugging Face das variáveis de ambiente
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.info("Por favor, adicione seu HF_TOKEN nas variáveis de ambiente (Secrets) para continuar.", icon="🔑")
else:
    # Componente de upload de arquivo
    uploaded_file = st.file_uploader("Escolha o arquivo do relatório (JPG ou PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.success("Arquivo carregado com sucesso!")
        
        # Exibe uma prévia da imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Prévia do documento enviado", use_column_width=True)

        if st.button("🧬 Iniciar Análise Documental Avançada"):
            with st.spinner("O modelo está processando o documento..."):
                try:
                    # Converte a imagem para bytes puros em formato JPEG
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Codifica em Base64 padrão para a API
                    base64_image = base64.b64encode(img_bytes).decode("utf-8")
                    
                    # Seu prompt altamente estruturado
                    prompt_texto = """
                    Você é um médico especialista em análise documental clínica, com amplo conhecimento em medicina baseada em evidências e classificação internacional de doenças (CID-10 e CID-11).
                    Sua tarefa é analisar cuidadosamente o relatório médico enviado na imagem em anexo.

                    Responda estruturando sua resposta rigorosamente com os tópicos pedidos anteriormente (Etapa 1 à Etapa 11), usando Markdown para formatação e tabelas.
                    """

                    # Usando a URL de inferência direta do modelo estável da comunidade Qwen
                    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    
                    # Monta o payload injetando a imagem como dados de contexto legíveis
                    payload = {
                        "inputs": f"{prompt_texto}\n\n[Dados da Imagem anexada para OCR]: data:image/jpeg;base64,{base64_image}",
                        "parameters": {
                            "max_new_tokens": 1500,
                            "return_full_text": False
                        }
                    }
                    
                    # Faz a requisição HTTP bruta
                    response = requests.post(API_URL, headers=headers, json=payload)
                    result = response.json()
                    
                    # Exibe o resultado na tela
                    st.markdown("---")
                    st.subheader("📑 Relatório de Análise Médica Gerado")
                    
                    if isinstance(result, list) and len(result) > 0:
                        st.markdown(result[0].get("generated_text", "Erro ao extrair o texto."))
                    elif "generated_text" in result:
                        st.markdown(result["generated_text"])
                    else:
                        # Se retornar erro de carregamento de modelo, mostra o aviso amigável do HF
                        if "estimated_time" in result:
                            st.warning(f"O modelo está acordando nos servidores do Hugging Face. Aguarde cerca de {int(result['estimated_time'])} segundos e clique no botão novamente.")
                        else:
                            st.error(f"Resposta inesperada da API: {result}")
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar a análise clínica: {e}")
