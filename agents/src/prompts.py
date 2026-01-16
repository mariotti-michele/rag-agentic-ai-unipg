from langchain.prompts import PromptTemplate

rag_prompt_template = """Sei un assistente accademico.
Hai accesso a estratti di documenti ufficiali.

Usa SOLO il contesto fornito per rispondere.
Se il contesto contiene il termine o l'argomento richiesto, indica che è presente e copia il testo più rilevante.
Non aggiungere nulla di tuo e non inventare.

Se davvero non ci sono riferimenti nemmeno parziali, rispondi esattamente: "Non presente nei documenti".

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.

Domanda: {question}

Contesto:
{context}

Risposta:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_prompt_template,
)

CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Sei un classificatore di query accademiche.

Devi scegliere una sola categoria tra:
- "semplice"
- "rag"

Rispondi SOLO con una di queste due parole.

Regole:
- Se la domanda contiene solo saluti, convenevoli o curiosità non universitarie (es. "ciao", "buongiorno", "come stai", "grazie", "che tempo fa", "chi sei") → rispondi: semplice
- In TUTTI gli altri casi, anche se la domanda è breve ma riguarda università, corsi, lezioni, orari, esami, tesi, lauree, tirocini, regolamenti, o informazioni accademiche → rispondi: rag

Domanda: {question}
Categoria:""",
)


simple_prompt_template = """Sei un assistente accademico gentile.
Rispondi in modo breve e diretto alla domanda generica seguente:

Domanda: {question}
Risposta:"""