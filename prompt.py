from langchain_core.prompts import PromptTemplate

# Define your custom prompt
custom_template = (
    "You are a helpful real estate assistant. "
    "Answer the following question based only on the provided context. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}"
)

# Create PromptTemplate
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_template,
)
