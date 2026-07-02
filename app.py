import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(page_title="Analisador Clínico Avançado", page_icon="🩺", layout="centered")

st.title("🩺 Analisador Clínico e Auditor de Relatórios Médicos")
st.write("Faça o upload do relatório (imagem ou PDF) para uma análise detalhada baseada em evidências.")

# Configuração da API do Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.info("Por favor, adicione sua GOOGLE_API_KEY nas variáveis de ambiente para continuar.", icon="🔑")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Inicializa o modelo Gemini 1.5 Flash (ideal para documentos complexos e tabelas)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Componente de upload de arquivo
    uploaded_file = st.file_uploader("Escolha o arquivo do relatório (JPG, PNG ou PDF)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        st.success("Arquivo carregado com sucesso!")
        
        # Exibe uma prévia se for imagem
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="Prévia do documento enviado", use_column_width=True)

        # Botão para iniciar a análise
        if st.button("🧬 Iniciar Análise Documental Avançada"):
            with st.spinner("A IA está realizando o OCR e aplicando as diretrizes clínicas..."):
                try:
                    # Preparando o arquivo para enviar para a API
                    bytes_data = uploaded_file.getvalue()
                    mime_type = uploaded_file.type
                    file_part = {
                        "mime_type": mime_type,
                        "data": bytes_data
                    }
                    
                    # Seu prompt altamente estruturado
                    prompt = """
                    Você é um médico especialista em análise documental clínica, com amplo conhecimento em medicina baseada em evidências e classificação internacional de doenças (CID-10 e CID-11).
                    Sua tarefa é analisar cuidadosamente o relatório médico enviado (inclusive se for proveniente de OCR ou imagem manuscrita).

                    Responda estritamente estruturando sua resposta com os seguintes tópicos em Markdown:

                    ### 🛡️ Etapa 1 – Validação da leitura
                    * **Grau de confiança da leitura:** (Alto, Médio ou Baixo)
                    * **Trechos ilegíveis, incompletos ou duvidosos:** * *Nota:* Nunca invente informações que não estejam claramente presentes no documento. Caso alguma informação seja incerta, informe isso explicitamente.

                    ### 📋 Etapa 2 – Resumo do relatório
                    * **Motivo da consulta:**
                    * **Principais sintomas:**
                    * **Histórico clínico relevante:**
                    * **Diagnósticos descritos:**
                    * **Exames citados:**
                    * **Condutas adotadas:**
                    * **Medicamentos prescritos:**
                    * **Recomendações médicas:**
                    * **Conclusão do relatório:**

                    ### 🔬 Etapa 3 – Análise clínica
                    1. **Diagnóstico principal:** (Explique qual parece ser a doença principal descrita)
                    2. **Diagnósticos secundários:** (Caso existam, liste-os)
                    3. **Classificação Internacional de Doenças (CID):**
                       * **CID-10:**
                       * **CID-11:** (quando aplicável)
                       * *Nota:* Caso o CID não esteja explícito, indique o CID mais provável e explique o motivo.

                    ### 📊 Etapa 4 – Evidências encontradas
                    (Liste quais informações do relatório sustentam o diagnóstico, como sintomas, exames, achados clínicos, histórico e evolução)

                    ### 💊 Etapa 5 – Tratamento recomendado atualmente
                    Com base nas diretrizes clínicas mais atuais e na medicina baseada em evidências, descreva:
                    * **Tratamento medicamentoso usual:**
                    * **Tratamentos não medicamentosos:**
                    * **Fisioterapia, psicoterapia ou reabilitação:** (quando indicados)
                    * **Cirurgia:** (quando indicada)
                    * **Exames complementares recomendados:**
                    * **Acompanhamento por especialistas:**
                    * **Prognóstico esperado:**
                    *(Caso existam diferentes opções terapêuticas, apresente todas)*

                    ### 🔄 Etapa 6 – Comparação
                    * **Status do tratamento no relatório:** (está atualizado / está parcialmente atualizado / está desatualizado)
                    * **Justificativa:** (Explique os motivos detalhadamente)

                    ### 📈 Etapa 7 – Prognóstico
                    * **Expectativa de evolução:**
                    * **Possibilidade de cura ou controle:**
                    * **Riscos de agravamento e possíveis complicações:**

                    ### 💰 Etapa 8 – Estimativa de custos do tratamento
                    Apresente uma estimativa aproximada considerando valores médios praticados no Brasil. Organize exatamente na tabela Markdown abaixo:

                    | Item | Frequência | Valor aproximado (R$) |
                    | :--- | :--- | :--- |
                    | Consultas | | |
                    | Exames | | |
                    | Medicamentos | | |
                    | Terapias | | |
                    | Procedimentos | | |

                    * **Custo mensal estimado:** R$ 
                    * **Custo anual estimado:** R$ 
                    * *Aviso de Custos:* Sempre informe que os valores são estimativas e podem variar conforme a região, convênio, rede pública ou privada e o tratamento efetivamente adotado.

                    ### ⚠️ Etapa 9 – Grau de gravidade
                    * **Classificação:** (Baixa / Moderada / Grave / Muito grave)
                    * **Justificativa técnica:**

                    ### 🎯 Etapa 10 – Pontos de atenção
                    (Liste inconsistências, informações ausentes, exames que ajudariam no diagnóstico e dúvidas relevantes)

                    ### 🏁 Etapa 11 – Conclusão
                    Faça uma conclusão em linguagem técnica, resumindo: doença principal, CID, situação clínica, tratamento indicado atualmente, prognóstico e estimativa de custos.

                    ---
                    **REGRAS OBRIGATÓRIAS DE RESPOSTA:**
                    1. Não invente informações. Diferencie claramente fatos presentes no relatório de inferências clínicas.
                    2. Quando houver incerteza, informe explicitamente.
                    3. Utilize linguagem técnica, objetiva e organizada.
                    4. Sempre indique que a conclusão DEPENDE de confirmação por avaliação médica presencial.
                    """
                    
                    # Chamada da API passando o novo prompt e o documento
                    response = model.generate_content([prompt, file_part])
                    
                    # Exibindo o resultado na tela do Streamlit
                    st.markdown("---")
                    st.subheader("📑 Relatório de Análise Médica Gerado")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar a análise clínica: {e}")