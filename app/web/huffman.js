/**
 * Implementação de Huffman Canônico em JavaScript (navegador e Node.js).
 * Deve concordar byte a byte com app/huffman.py (formato descrito no cabeçalho de lá).
 */

function _heapPush(heap, item) {
  heap.push(item);
  let i = heap.length - 1;
  while (i > 0) {
    const pai = (i - 1) >> 1;
    const a = heap[i];
    const b = heap[pai];
    if (a.freq < b.freq || (a.freq === b.freq && a.ordem < b.ordem)) {
      heap[i] = b;
      heap[pai] = a;
      i = pai;
    } else {
      break;
    }
  }
}

function _heapPop(heap) {
  const topo = heap[0];
  const ultimo = heap.pop();
  if (heap.length > 0) {
    heap[0] = ultimo;
    let i = 0;
    const n = heap.length;
    while (true) {
      let menor = i;
      const esq = 2 * i + 1;
      const dir = 2 * i + 2;

      if (
        esq < n &&
        (heap[esq].freq < heap[menor].freq ||
          (heap[esq].freq === heap[menor].freq && heap[esq].ordem < heap[menor].ordem))
      ) {
        menor = esq;
      }
      if (
        dir < n &&
        (heap[dir].freq < heap[menor].freq ||
          (heap[dir].freq === heap[menor].freq && heap[dir].ordem < heap[menor].ordem))
      ) {
        menor = dir;
      }
      if (menor === i) break;

      const tmp = heap[i];
      heap[i] = heap[menor];
      heap[menor] = tmp;
      i = menor;
    }
  }
  return topo;
}

/**
 * Calcula o comprimento (em bits) do código Huffman de cada byte (0..255).
 * @param {Uint8Array} dados
 * @returns {Uint8Array}
 */
function calcularComprimentos(dados) {
  const freqs = new Uint32Array(256);
  for (let i = 0; i < dados.length; i++) {
    freqs[dados[i]]++;
  }

  const heap = [];
  let ordem = 0;
  for (let b = 0; b < 256; b++) {
    if (freqs[b] > 0) {
      _heapPush(heap, { freq: freqs[b], ordem: ordem++, no: b });
    }
  }

  const comprimentos = new Uint8Array(256);
  if (heap.length === 0) {
    return comprimentos;
  }

  // Caso de canto: apenas 1 símbolo distinto nos dados.
  if (heap.length === 1) {
    comprimentos[heap[0].no] = 1;
    return comprimentos;
  }

  while (heap.length > 1) {
    const n1 = _heapPop(heap);
    const n2 = _heapPop(heap);
    _heapPush(heap, {
      freq: n1.freq + n2.freq,
      ordem: ordem++,
      no: [n1.no, n2.no],
    });
  }

  const raiz = heap[0].no;
  const pilha = [[raiz, 0]];
  while (pilha.length > 0) {
    const [no, prof] = pilha.pop();
    if (typeof no === "number") {
      comprimentos[no] = prof;
    } else {
      pilha.push([no[0], prof + 1]);
      pilha.push([no[1], prof + 1]);
    }
  }

  return comprimentos;
}

/**
 * Gera a tabela canônica de códigos a partir dos 256 comprimentos.
 * Retorna um array de 256 posições onde cada item presente é { codigo, tam }.
 * @param {Uint8Array} comprimentos
 * @returns {Array<{codigo: number, tam: number} | null>}
 */
function gerarCodigosCanonicos(comprimentos) {
  const simbolos = [];
  for (let b = 0; b < 256; b++) {
    if (comprimentos[b] > 0) {
      simbolos.push({ tam: comprimentos[b], byteVal: b });
    }
  }

  const codigos = new Array(256).fill(null);
  if (simbolos.length === 0) {
    return codigos;
  }

  // Ordenação canônica: comprimento crescente, depois valor do byte crescente.
  simbolos.sort((a, b) => a.tam - b.tam || a.byteVal - b.byteVal);

  let codigo = 0;
  let tamAnterior = simbolos[0].tam;

  for (let i = 0; i < simbolos.length; i++) {
    const { tam, byteVal } = simbolos[i];
    if (i > 0) {
      codigo = ((codigo + 1) << (tam - tamAnterior)) >>> 0;
    }
    codigos[byteVal] = { codigo, tam };
    tamAnterior = tam;
  }

  return codigos;
}

/**
 * Compacta um Uint8Array usando Huffman Canônico.
 * @param {Uint8Array} dados
 * @returns {Uint8Array}
 */
function compactar(dados) {
  if (dados.length === 0) {
    return new Uint8Array(260);
  }

  const comprimentos = calcularComprimentos(dados);
  const codigos = gerarCodigosCanonicos(comprimentos);

  const corpo = [];
  let buffer = 0;
  let bitsNoBuffer = 0;

  for (let i = 0; i < dados.length; i++) {
    const { codigo, tam } = codigos[dados[i]];
    buffer = ((buffer << tam) | codigo) >>> 0;
    bitsNoBuffer += tam;

    while (bitsNoBuffer >= 8) {
      bitsNoBuffer -= 8;
      corpo.push((buffer >>> bitsNoBuffer) & 0xff);
      buffer &= (1 << bitsNoBuffer) - 1;
    }
  }

  if (bitsNoBuffer > 0) {
    corpo.push((buffer << (8 - bitsNoBuffer)) & 0xff);
  }

  const saida = new Uint8Array(260 + corpo.length);
  const view = new DataView(saida.buffer);
  view.setUint32(0, dados.length, false); // false = big-endian
  saida.set(comprimentos, 4);
  saida.set(corpo, 260);

  return saida;
}

/**
 * Descompacta um Uint8Array gerado no formato Huffman Canônico.
 * @param {Uint8Array} payload
 * @returns {Uint8Array}
 */
function descompactar(payload) {
  if (payload.length < 260) {
    throw new Error("Payload inválido: menor que o cabeçalho mínimo de 260 bytes.");
  }

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const tamanhoOriginal = view.getUint32(0, false); // false = big-endian
  if (tamanhoOriginal === 0) {
    return new Uint8Array(0);
  }

  const comprimentos = payload.subarray(4, 260);
  const codigos = gerarCodigosCanonicos(comprimentos);

  let temCodigo = false;
  const raiz = [null, null];
  for (let b = 0; b < 256; b++) {
    const entrada = codigos[b];
    if (!entrada) continue;
    temCodigo = true;

    const { codigo, tam } = entrada;
    let no = raiz;
    for (let i = tam - 1; i >= 0; i--) {
      const bit = (codigo >>> i) & 1;
      if (i === 0) {
        if (no[bit] !== null) {
          throw new Error("Tabela Huffman inválida: conflito de prefixo.");
        }
        no[bit] = b;
      } else {
        if (no[bit] === null) {
          no[bit] = [null, null];
        } else if (!Array.isArray(no[bit])) {
          throw new Error("Tabela Huffman inválida: conflito de prefixo.");
        }
        no = no[bit];
      }
    }
  }

  if (!temCodigo) {
    throw new Error("Payload inválido: tamanho > 0 sem tabela de códigos.");
  }

  const saida = new Uint8Array(tamanhoOriginal);
  let posSaida = 0;
  let no = raiz;

  for (let p = 260; p < payload.length; p++) {
    const byteComp = payload[p];
    for (let i = 7; i >= 0; i--) {
      const bit = (byteComp >>> i) & 1;
      const proximo = no[bit];
      if (proximo === null) {
        throw new Error("Fluxo de bits inválido para a tabela Huffman informada.");
      }
      if (typeof proximo === "number") {
        saida[posSaida++] = proximo;
        if (posSaida === tamanhoOriginal) {
          return saida;
        }
        no = raiz;
      } else {
        no = proximo;
      }
    }
  }

  throw new Error("Dados compactados truncados antes de atingir o tamanho original.");
}

const Huffman = {
  calcularComprimentos,
  gerarCodigosCanonicos,
  compactar,
  descompactar,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = Huffman;
}
if (typeof window !== "undefined") {
  window.Huffman = Huffman;
}
