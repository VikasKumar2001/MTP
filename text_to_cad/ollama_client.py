import numpy as np
import ollama

MODEL_NAME = "nomic-embed-text"


def get_text_embedding(text: str) -> np.ndarray:
    """
    Convert a natural-language query into an Ollama text embedding.

    Parameters
    ----------
    text : str
        Natural-language CAD search query.

    Returns
    -------
    np.ndarray
        1-D floating-point embedding vector.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")

    try:
        response = ollama.embed(model=MODEL_NAME, input=text)
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Ollama. Make sure Ollama is running "
            f"and the model '{MODEL_NAME}' is installed."
        ) from e

    embeddings = response["embeddings"]
    if not embeddings:
        raise RuntimeError("Ollama returned an empty embedding.")

    # Ollama returns a list of embeddings; take the first one
    vector = np.asarray(embeddings[0], dtype=np.float32)
    return vector


if __name__ == "__main__":
    query = "Find a comfortable desk chair."
    embedding = get_text_embedding(query)

    print("=" * 60)
    print("Ollama Text Embedding Test")
    print("=" * 60)
    print("Model      :", MODEL_NAME)
    print("Query      :", query)
    print("Dimension  :", embedding.shape[0])
    print("Data type  :", embedding.dtype)
    print("\nFirst 10 values:")
    print(embedding[:10])
    print("\nL2 norm:")
    print(np.linalg.norm(embedding))
    print("=" * 60)