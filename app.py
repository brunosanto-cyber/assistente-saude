import streamlit as st
import os
import io
import base64
from PIL import Image
from huggingface_hub import InferenceClient

# Configuração da página do Streamlit
st.set_page_config(page_title="Analisador Clínico Avançado", page_icon="🩺", layout="centered")

st.title("🩺 Analisador Clínico e Auditor de Relatórios Médicos (HF)")
st.write("Faça o upload do relatório para uma análise detalhada baseada em evidências.")

# Pega o Token do Hugging Face das variáveis de ambiente
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.info("Por favor, adicione seu HF_TOKEN nas variáveis de ambiente para continuar.", icon="🔑")
else:
    # Inicializa o cliente oficial do Hugging Face
    client = InferenceClient(token=HF_TOKEN)

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
                    # Converte a imagem para bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Converte os bytes para texto codificado em Base64
                    base64_image = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{base64_image}"
                    
                    # Seu prompt altamente estruturado
                    prompt_texto = """
                    Você é um médico especialista em análise documental clínica, com amplo conhecimento em medicina baseada em evidências e classificação internacional de doenças (CID-10 e CID-11).
                    Sua tarefa é analisar cuidadosamente o relatório médico enviado na imagem.

                    Responda estruturando sua resposta rigorosamente com os tópicos pedidos anteriormente (Etapa 1 à Etapa 11), usando Markdown para formatação e tabelas.
                    """

                    # Chamada utilizando o modelo Qwen2.5-VL (Livre de bloqueios 403)
                    response = client.chat_completion(
                        model="Qwen/Qwen2.5-VL-7B-Instruct",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_texto},
                                    {"type": "image_url", "image_url": {"url": data_url}}
                                ]
                            }
                        ],
                        max_tokens=2000
                    )
                    
                    # Exibe o resultado na tela
                    st.markdown("---")
                    st.subheader("📑 Relatório de Análise Médica Gerado")
                    
                    texto_resposta = response.choices[0].message.content
                    st.markdown(texto_resposta)
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar a análise clínica: {e}")
