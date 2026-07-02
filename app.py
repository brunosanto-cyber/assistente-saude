import streamlit as st
import os
import io
from PIL import Image
from huggingface_hub import InferenceClient

# Configuração da página do Streamlit
st.set_page_config(page_title="Analisador Clínico Avançado", page_icon="🩺", layout="centered")

st.title("🩺 Analisador Clínico e Auditor de Relatórios Médicos (HF)")
st.write("Faça o upload do relatório para uma análise detalhada baseada em evidências.")

# Pega o Token do Hugging Face das variáveis de ambiente
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.info("Por favor, adicione seu HF_TOKEN nas variáveis de ambiente (Secrets) para continuar.", icon="🔑")
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
                    # Converte a imagem para bytes puros em formato JPEG
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Seu prompt altamente estruturado
                    prompt_texto = """
                    Você é um médico especialista em análise documental clínica, com amplo conhecimento em medicina baseada em evidências e classificação internacional de doenças (CID-10 e CID-11).
                    Sua tarefa é analisar cuidadosamente o relatório médico enviado na imagem.

                    Responda estruturando sua resposta rigorosamente com os tópicos pedidos anteriormente (Etapa 1 à Etapa 11), usando Markdown para formatação e tabelas.
                    """

                    # Rota alternativa de visão computacional direta (Ignora o bloqueio 403 de Chat)
                    resposta_texto = client.visual_question_answering(
                        image=img_bytes,
                        question=prompt_texto,
                        model="dandelin/vilt-b32-finetuned-vqa" # Modelo público e aberto para leitura de imagens
                    )
                    
                    # Exibe o resultado na tela
                    st.markdown("---")
                    st.subheader("📑 Relatório de Análise Médica Gerado")
                    
                    # Trata o retorno da API
                    if isinstance(resposta_texto, list) and len(resposta_texto) > 0:
                        st.markdown(resposta_texto[0].get("answer", resposta_texto))
                    elif hasattr(resposta_texto, "answer"):
                        st.markdown(resposta_texto.answer)
                    else:
                        st.markdown(resposta_texto)
                        
                except Exception as e:
                    # Caso o modelo VQA falhe, tenta o fallback de geração de texto pura passando a imagem descrita
                    st.warning("Tentando rota secundária de processamento...")
                    try:
                        # Fallback enviando a imagem diretamente para a API de geração de imagem-para-texto
                        output = client.image_to_text(img_bytes)
                        st.markdown(output)
                    except Exception as erro_secundario:
                        st.error(f"Ocorreu um erro ao processar a análise clínica: {e}")
