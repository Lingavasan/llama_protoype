from src.memory_architect.core.config import SystemSettings

def test_config_loading():
    print("Attempting to load SystemSettings...")
    settings = SystemSettings()
    
    print("Settings loaded successfully.")
    print(f"Environment: {settings.environment}")
    print(f"LLM Model Path: {settings.models.llm_model_path}")
    print(f"Retention Policy: {settings.retention.policy_name}")
    print(f"PII Entities: {settings.privacy.pii_entities}")
    
    assert settings.models.llm_model_path == "models/llama-2-7b-chat.gguf"
    assert settings.retention.policy_name == "Standard_Agent_Policy"
    assert "US_SSN" in settings.privacy.pii_entities
    print("All assertions passed.")

if __name__ == "__main__":
    test_config_loading()
