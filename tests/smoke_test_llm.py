import os
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def test_llm_inference():
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=200,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=2048,
    )
    
    print("\nModel loaded. Running inference...")
    prompt = "Q: What is the primary function of human memory? A:"
    response = llm.invoke(prompt)
    
    print("\n\nInference complete.")
    print("-" * 20)
    print(response)
    print("-" * 20)

if __name__ == "__main__":
    test_llm_inference()
