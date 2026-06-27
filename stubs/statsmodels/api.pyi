from pandas import DataFrame

class _Stats:
    def anova_lm(self, *results: object, typ: int = ...) -> DataFrame: ...

stats: _Stats
