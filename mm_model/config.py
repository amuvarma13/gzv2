from transformers import PretrainedConfig, CONFIG_MAPPING

class OrpheusConfig(PretrainedConfig):
    
    model_type = "gazelle"

    is_composition = False
    def __init__(
        self,
        audio_config=None,
        text_config=None,
        text_model_id=None,
        ignore_index=-100,
        audio_token_index=32000,
        vocab_size=32000,
        hidden_size=3072,
        stack_factor=8,
        projector_type="mlp",
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index
        self.vocab_size = vocab_size

        self.text_model_id = text_model_id

        self.audio_config = audio_config
        self.text_config = text_config

        self.hidden_size = hidden_size
        print("self.hidden_size", self.hidden_size)
        self.stack_factor = stack_factor
        self.projector_type = projector_type

        if isinstance(self.text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()
        
        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "wav2vec2"
            )
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
            self.vocab_size = self.audio_config.vocab_size
        elif audio_config is None:
            self.audio_config = CONFIG_MAPPING["wav2vec2"]()

        super().__init__(**kwargs)
