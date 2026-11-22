"""Huffman Canonico: compactacao ponta-a-ponta dos arquivos enviados/recebidos.

Formato (identico a app/web/huffman.js, byte a byte):
  - 4 bytes: tamanho original, uint32 big-endian
  - 256 bytes: comprimento (em bits) do codigo de cada byte 0..255 (0 = ausente)
  - corpo: codigos canonicos concatenados, MSB primeiro, ultimo byte completado com zeros
Codigos canonicos: simbolos ordenados por (comprimento, valor do byte).
"""
import heapq
import struct

HEADER_SIZE = 260


def tamanho_declarado(payload: bytes) -> int:
    """Tamanho original declarado no cabecalho (sem descompactar)."""
    if len(payload) < 4:
        raise ValueError("Payload inválido: menor que o cabeçalho mínimo de 260 bytes.")
    return struct.unpack(">I", payload[:4])[0]


def validar_cabecalho(payload: bytes) -> int:
    """Confere o cabecalho sem descompactar e devolve o tamanho original declarado.

    A tabela de comprimentos precisa formar um codigo de prefixo completo
    (soma de Kraft == 1), ou ter um unico simbolo de 1 bit. Isso recusa logo
    arquivos que nao foram compactados com este formato (ex.: CSV em claro).
    """
    tamanho = tamanho_declarado(payload)
    if len(payload) < HEADER_SIZE:
        raise ValueError("Payload inválido: menor que o cabeçalho mínimo de 260 bytes.")
    comprimentos = [c for c in payload[4:HEADER_SIZE] if c > 0]
    if tamanho == 0:
        return 0
    if not comprimentos:
        raise ValueError("Payload inválido: tamanho > 0 sem tabela de códigos.")
    if comprimentos == [1]:
        return tamanho
    maior = max(comprimentos)
    if sum(1 << (maior - c) for c in comprimentos) != 1 << maior:
        raise ValueError("Tabela Huffman inválida: não é um código de prefixo completo.")
    return tamanho


def _calcular_comprimentos(dados: bytes) -> list[int]:
    """Calcula o comprimento (em bits) do código Huffman de cada byte (0..255)."""
    freqs = [0] * 256
    for b in dados:
        freqs[b] += 1

    # Cada item da fila é (frequencia, ordem_insercao, no).
    # O campo 'ordem_insercao' evita que o heapq compare nós (int vs tuple) em empates.
    heap: list[tuple[int, int, int | tuple]] = []
    ordem = 0
    for byte_val, freq in enumerate(freqs):
        if freq > 0:
            heap.append((freq, ordem, byte_val))
            ordem += 1

    heapq.heapify(heap)

    comprimentos = [0] * 256
    if not heap:
        return comprimentos

    # Caso de canto: apenas 1 símbolo distinto nos dados.
    if len(heap) == 1:
        _, _, unico_byte = heap[0]
        assert isinstance(unico_byte, int)
        comprimentos[unico_byte] = 1
        return comprimentos

    # Constrói a árvore de Huffman combinando sempre os dois nós de menor frequência.
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, ordem, (n1, n2)))
        ordem += 1

    raiz = heap[0][2]

    # Percorre a árvore de forma iterativa para registrar a profundidade das folhas.
    pilha: list[tuple[int | tuple, int]] = [(raiz, 0)]
    while pilha:
        no, prof = pilha.pop()
        if isinstance(no, int):
            comprimentos[no] = prof
        else:
            esq, dir_ = no
            pilha.append((esq, prof + 1))
            pilha.append((dir_, prof + 1))

    return comprimentos


def _gerar_codigos_canonicos(comprimentos: list[int]) -> dict[int, tuple[int, int]]:
    """Gera o mapa {byte: (codigo_int, comprimento)} pela regra canônica."""
    simbolos = sorted(
        (tam, byte_val)
        for byte_val, tam in enumerate(comprimentos)
        if tam > 0
    )
    if not simbolos:
        return {}

    codigos: dict[int, tuple[int, int]] = {}
    codigo = 0
    tam_anterior = simbolos[0][0]

    for i, (tam, byte_val) in enumerate(simbolos):
        if i > 0:
            codigo = (codigo + 1) << (tam - tam_anterior)
        codigos[byte_val] = (codigo, tam)
        tam_anterior = tam

    return codigos


def compactar(dados: bytes) -> bytes:
    """Compacta uma sequência de bytes usando Huffman Canônico."""
    if not dados:
        return bytes(HEADER_SIZE)

    comprimentos = _calcular_comprimentos(dados)
    codigos = _gerar_codigos_canonicos(comprimentos)

    # Cabeçalho: 4 bytes (tamanho original big-endian) + 256 bytes (comprimentos).
    saida = bytearray(struct.pack(">I", len(dados)))
    saida.extend(comprimentos)

    buffer = 0
    bits_no_buffer = 0

    for b in dados:
        codigo, tam = codigos[b]
        buffer = (buffer << tam) | codigo
        bits_no_buffer += tam

        while bits_no_buffer >= 8:
            bits_no_buffer -= 8
            saida.append((buffer >> bits_no_buffer) & 0xFF)
            buffer &= (1 << bits_no_buffer) - 1

    # Padding com zeros à direita (LSB) caso o último byte fique incompleto.
    if bits_no_buffer > 0:
        saida.append((buffer << (8 - bits_no_buffer)) & 0xFF)

    return bytes(saida)


def descompactar(payload: bytes) -> bytes:
    """Descompacta um payload gerado no formato Huffman Canônico."""
    if len(payload) < HEADER_SIZE:
        raise ValueError("Payload inválido: menor que o cabeçalho mínimo de 260 bytes.")

    tamanho_original = struct.unpack(">I", payload[:4])[0]
    if tamanho_original == 0:
        return b""

    comprimentos = list(payload[4:HEADER_SIZE])
    codigos = _gerar_codigos_canonicos(comprimentos)
    if not codigos:
        raise ValueError("Payload inválido: tamanho > 0 sem tabela de códigos.")

    # Monta a árvore de decodificação: nós internos são [filho_0, filho_1], folhas são int.
    raiz: list = [None, None]
    for byte_val, (codigo, tam) in codigos.items():
        no = raiz
        for i in range(tam - 1, -1, -1):
            bit = (codigo >> i) & 1
            if i == 0:
                if no[bit] is not None:
                    raise ValueError("Tabela Huffman inválida: conflito de prefixo.")
                no[bit] = byte_val
            else:
                if no[bit] is None:
                    no[bit] = [None, None]
                elif not isinstance(no[bit], list):
                    raise ValueError("Tabela Huffman inválida: conflito de prefixo.")
                no = no[bit]

    saida = bytearray()
    no = raiz
    for byte_comp in payload[HEADER_SIZE:]:
        for i in range(7, -1, -1):
            bit = (byte_comp >> i) & 1
            proximo = no[bit]
            if proximo is None:
                raise ValueError("Fluxo de bits inválido para a tabela Huffman informada.")
            if isinstance(proximo, int):
                saida.append(proximo)
                if len(saida) == tamanho_original:
                    return bytes(saida)
                no = raiz
            else:
                no = proximo

    raise ValueError("Dados compactados truncados antes de atingir o tamanho original.")
