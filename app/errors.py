"""Erros de dominio. Cada um carrega o status HTTP correspondente."""


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CsvValidationError(AppError):
    status_code = 422


class PayloadTooLargeError(AppError):
    status_code = 413


class NoModelError(AppError):
    status_code = 409


class BusyError(AppError):
    status_code = 409


class CompressionError(AppError):
    """Payload Huffman malformado ou corrompido."""
    status_code = 400
