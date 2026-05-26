class TokenBudgetManager:
    CHARS_PER_TOKEN = 3.5

    def __init__(self, n_ctx=4096, system_reserve=512, response_reserve=1024):
        self.n_ctx = n_ctx
        self.system_reserve = system_reserve
        self.response_reserve = response_reserve

    @property
    def context_budget(self):
        return self.n_ctx - self.system_reserve - self.response_reserve

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def trim_context(self, code_context: str, prompt: str, system_prompt: str):
        sys_t = self.estimate_tokens(system_prompt)
        pr_t  = self.estimate_tokens(prompt)
        available = self.context_budget - sys_t - pr_t
        
        if available <= 0:
            return "", True
            
        if self.estimate_tokens(code_context) <= available:
            return code_context, False
            
        return code_context[:int(available * self.CHARS_PER_TOKEN)], True

    def get_usage_pct(self, code_context, prompt, system_prompt):
        total = (self.estimate_tokens(system_prompt) +
                 self.estimate_tokens(prompt) +
                 self.estimate_tokens(code_context))
        return min(100, int(total / self.n_ctx * 100))