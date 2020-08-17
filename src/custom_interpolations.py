from omegaconf import OmegaConf


OmegaConf.register_resolver("times", lambda a,b: int(a) * int(b))
OmegaConf.register_resolver("times_divide", lambda a,b,c: int(float(a) * float(b) / float(c)))
OmegaConf.register_resolver("plus", lambda a,b: int(a) + int(b))
