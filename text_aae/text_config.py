class TextConfig(object):
    def __init__(
            self,
            model_fn,
            input_fns,
            mode='rnn'):
        self.model_fn = model_fn
        self.input_fns = input_fns
        self.mode = mode
