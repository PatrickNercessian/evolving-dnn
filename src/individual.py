class Individual:
    def __init__(self, id: int, **kwargs):
        self.id = id
        self.fitness = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"Individual(id={self.id})"

    def __repr__(self):
        return self.__str__()
