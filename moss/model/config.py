class GenerationConfig:

    def __init__(self):
        self._from_model_config = True
        self.bos_token_id = 150004
        self.eos_token_id = 150005
        self.pad_token_id = 0
        self.transformers_version = "4.26.1"
        self.do_sample = True
        self.max_length = 2048
        self.temperature = 0.95
        self.top_p = 0.7

        self.max_new_tokens = None
        self.num_beams = 1
        self.top_k = None
        self.typical_p = None
        self.epsilon_cutoff = None
        self.eta_cutoff = None
        self.renormalize_logits = None
    
    def update(self, **kw):
        res = {k:v for k, v in self.__dict__.items()}
        res.update(**kw)
        return res


if __name__ == "__main__":
    c = GenerationConfig()
    print(c.update(d=0))
