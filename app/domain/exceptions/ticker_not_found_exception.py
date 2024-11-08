class TickerNotFoundException(Exception):
    def __init__(self, detail: str = "No se encontró data histórica para la acción."):
        self.detail = detail