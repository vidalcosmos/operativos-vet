#!/usr/bin/env python3
"""
Extrae dia, mes y lugar de cada flyer de operativo veterinario usando Claude con vision.
El año se obtiene de la fecha de modificacion del archivo.
Procesa TODAS las imagenes de cada carpeta "operativos N" y genera un index.html buscable.
"""

import re
import base64
import json
import io
from datetime import datetime
from pathlib import Path
import anthropic
from PIL import Image

# -- Configuracion ------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
CACHE_FILE  = BASE_DIR / "operativos_cache.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_BYTES   = 4 * 1024 * 1024   # 4 MB — margen bajo el limite de 5 MB de la API

MESES_ORDEN = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# -- Helpers ------------------------------------------------------------------

def todas_las_imagenes(carpeta: Path) -> list[Path]:
    return sorted(f for f in carpeta.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def imagen_a_base64(ruta: Path) -> tuple[str, str]:
    """Convierte imagen a base64, comprimiendo con Pillow si supera el limite."""
    raw = ruta.read_bytes()

    if len(raw) <= MAX_BYTES:
        media_types = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
        }
        return base64.standard_b64encode(raw).decode(), media_types.get(ruta.suffix.lower(), "image/jpeg")

    img = Image.open(ruta).convert("RGB")
    quality = 85
    while True:
        buf = io.BytesIO()
        max_side = 2048 if quality > 50 else 1280
        img_r = img.copy()
        img_r.thumbnail((max_side, max_side), Image.LANCZOS)
        img_r.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= MAX_BYTES or quality <= 30:
            break
        quality -= 15

    return base64.standard_b64encode(data).decode(), "image/jpeg"


def extraer_info(cliente: anthropic.Anthropic, imagen_path: Path) -> dict:
    """
    Llama a Claude con vision para extraer dia, mes y lugar del flyer.
    Devuelve un dict: {"dia": str, "mes": str, "lugar": str}
    """
    b64, media_type = imagen_a_base64(imagen_path)

    prompt = """\
Este es un flyer de un operativo veterinario comunitario.
Extrae la siguiente informacion que aparece escrita en la imagen y respondeme UNICAMENTE con un JSON valido, sin texto adicional:

{
  "dia": "<numero del dia del evento, solo el numero, ej: 5>",
  "mes": "<nombre del mes en español con la primera letra en mayuscula, ej: Marzo>",
  "lugar": "<nombre del lugar o direccion donde se realiza el operativo>"
}

Reglas:
- Si hay varias ubicaciones en el lugar, separalas con ' / '.
- Si no puedes leer un campo, usa null para ese campo.
- No incluyas nada fuera del JSON."""

    respuesta = cliente.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    texto = ""
    for bloque in respuesta.content:
        if bloque.type == "text":
            texto = bloque.text.strip()
            break

    # Parsear JSON — extraer el bloque {} aunque haya texto extra
    try:
        inicio = texto.index("{")
        fin    = texto.rindex("}") + 1
        datos  = json.loads(texto[inicio:fin])
        return {
            "dia":   str(datos.get("dia") or "").strip() or "?",
            "mes":   str(datos.get("mes") or "").strip() or "?",
            "lugar": str(datos.get("lugar") or "").strip() or "No especificado",
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return {"dia": "?", "mes": "?", "lugar": texto or "No especificado"}


def cargar_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def es_cache_valido(valor) -> bool:
    """El cache nuevo guarda dicts con dia/mes/lugar. Los strings del cache viejo son invalidos."""
    return isinstance(valor, dict) and "dia" in valor and "mes" in valor and "lugar" in valor


# -- Procesamiento principal --------------------------------------------------

def procesar_carpetas() -> list[dict]:
    cliente = anthropic.Anthropic()
    cache   = cargar_cache()

    patron = re.compile(r"^operativos\s+(\d+)$", re.IGNORECASE)
    carpetas = sorted(
        [(int(m.group(1)), e)
         for e in BASE_DIR.iterdir()
         if e.is_dir() and (m := patron.match(e.name))],
        key=lambda x: x[0],
    )

    resultados     = []
    total_imagenes = 0
    cache_hits     = 0
    reprocesadas   = 0

    for numero, carpeta in carpetas:
        imagenes = todas_las_imagenes(carpeta)
        if not imagenes:
            print(f"  [{numero:02d}] {carpeta.name} — sin imagenes, omitiendo.")
            continue

        print(f"  [{numero:02d}] {carpeta.name} — {len(imagenes)} imagen(es)")

        for idx, imagen in enumerate(imagenes, start=1):
            cache_key = str(imagen.relative_to(BASE_DIR))
            anio = datetime.fromtimestamp(imagen.stat().st_mtime).year

            if cache_key in cache and es_cache_valido(cache[cache_key]):
                info = cache[cache_key]
                print(f"        {idx:>3}/{len(imagenes)} {imagen.name[:45]:<45} -> (cache) {info['dia']} {info['mes']} | {info['lugar']}")
                cache_hits += 1
            else:
                print(f"        {idx:>3}/{len(imagenes)} {imagen.name[:45]:<45} -> procesando...", end=" ", flush=True)
                info = extraer_info(cliente, imagen)
                cache[cache_key] = info
                guardar_cache(cache)
                print(f"{info['dia']} {info['mes']} | {info['lugar']}")
                reprocesadas += 1

            resultados.append({
                "numero":  numero,
                "carpeta": carpeta.name,
                "imagen":  imagen.name,
                "dia":     info["dia"],
                "mes":     info["mes"],
                "anio":    anio,
                "lugar":   info["lugar"],
                # Clave numerica para ordenar por fecha
                "mes_num": MESES_ORDEN.get(info["mes"].lower(), 99),
            })
            total_imagenes += 1

    print(f"\nTotal: {len(carpetas)} carpetas, {total_imagenes} imagenes")
    print(f"  Cache hits: {cache_hits} | Nuevas/reprocesadas: {reprocesadas}")
    return resultados


# -- Generacion de HTML -------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Operativos Veterinarios</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f4f8; margin: 0; padding: 24px 16px; color: #1a202c;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; }}

    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 4px; color: #2d3748; }}
    .subtitle {{ font-size: 0.9rem; color: #718096; margin-bottom: 24px; }}

    .search-wrap {{ position: relative; margin-bottom: 20px; }}
    .search-wrap svg {{
      position: absolute; left: 12px; top: 50%;
      transform: translateY(-50%); color: #a0aec0;
    }}
    #buscar {{
      width: 100%; padding: 10px 12px 10px 40px; font-size: 1rem;
      border: 1px solid #cbd5e0; border-radius: 8px; outline: none;
      transition: border-color .2s, box-shadow .2s; background: #fff;
    }}
    #buscar:focus {{
      border-color: #4299e1; box-shadow: 0 0 0 3px rgba(66,153,225,.25);
    }}

    #contador {{ font-size: 0.85rem; color: #718096; margin-bottom: 12px; }}

    .tabla-wrap {{
      overflow-x: auto; border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.08);
    }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}

    thead th {{
      background: #2b6cb0; color: #fff; padding: 12px 16px;
      text-align: left; font-size: 0.82rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: .05em;
      cursor: pointer; user-select: none; white-space: nowrap;
    }}
    thead th:hover {{ background: #2c5282; }}
    thead th .arrow {{ display: inline-block; margin-left: 5px; opacity: .5; }}
    thead th.asc  .arrow::after {{ content: "\\25B2"; opacity: 1; }}
    thead th.desc .arrow::after {{ content: "\\25BC"; opacity: 1; }}
    thead th:not(.asc):not(.desc) .arrow::after {{ content: "\\2B0D"; }}

    tbody tr {{ border-bottom: 1px solid #e2e8f0; transition: background .15s; }}
    tbody tr:last-child {{ border-bottom: none; }}
    tbody tr:hover {{ background: #ebf8ff; }}
    tbody tr.oculta {{ display: none; }}
    tbody tr.grupo-par   {{ background: #fff; }}
    tbody tr.grupo-par:hover   {{ background: #ebf8ff; }}
    tbody tr.grupo-impar {{ background: #f7fafc; }}
    tbody tr.grupo-impar:hover {{ background: #ebf8ff; }}

    td {{ padding: 10px 16px; font-size: 0.93rem; vertical-align: middle; }}

    td.dia-col  {{ text-align: center; font-weight: 700; color: #2b6cb0; width: 52px; }}
    td.mes-col  {{ font-weight: 600; color: #2b6cb0; white-space: nowrap; }}
    td.anio-col {{ text-align: center; color: #718096; font-size: 0.88rem; width: 64px; }}
    td.lugar-col {{ }}

    .sin-datos {{ color: #a0aec0; font-style: italic; }}

    .no-resultados {{
      text-align: center; padding: 40px 16px; color: #a0aec0; font-size: 1rem;
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>Operativos Veterinarios</h1>
  <p class="subtitle">Indice completo &mdash; {total} operativos en {carpetas} carpetas</p>

  <div class="search-wrap">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    <input id="buscar" type="search" placeholder="Buscar por dia, mes, año o lugar..." autocomplete="off" />
  </div>

  <p id="contador">{total} operativos</p>

  <div class="tabla-wrap">
    <table id="tabla">
      <thead>
        <tr>
          <th data-col="0">Dia<span class="arrow"></span></th>
          <th data-col="1" class="asc">Mes<span class="arrow"></span></th>
          <th data-col="2">Año<span class="arrow"></span></th>
          <th data-col="3">Lugar<span class="arrow"></span></th>
        </tr>
      </thead>
      <tbody>
{filas}
      </tbody>
    </table>
    <p class="no-resultados" id="sinResultados" style="display:none;">
      No se encontraron resultados.
    </p>
  </div>
</div>

<script>
(function () {{
  const input   = document.getElementById('buscar');
  const tabla   = document.getElementById('tabla');
  const counter = document.getElementById('contador');
  const sinRes  = document.getElementById('sinResultados');
  const ths     = tabla.querySelectorAll('thead th');
  let sortCol   = 1, sortAsc = true;

  function getFilas() {{ return Array.from(tabla.querySelectorAll('tbody tr')); }}

  function actualizar() {{
    const q = input.value.toLowerCase().trim();
    let visible = 0;
    getFilas().forEach(tr => {{
      const mostrar = !q || tr.dataset.busqueda.includes(q);
      tr.classList.toggle('oculta', !mostrar);
      if (mostrar) visible++;
    }});
    counter.textContent = visible + ' operativo' + (visible !== 1 ? 's' : '');
    sinRes.style.display = visible === 0 ? '' : 'none';
  }}

  function ordenar(col) {{
    sortAsc = sortCol === col ? !sortAsc : true;
    sortCol = col;
    ths.forEach((th, i) => {{
      th.classList.remove('asc', 'desc');
      if (i === col) th.classList.add(sortAsc ? 'asc' : 'desc');
    }});
    const tbody = tabla.querySelector('tbody');
    getFilas()
      .sort((a, b) => {{
        const va = a.children[col].dataset.val ?? a.children[col].textContent;
        const vb = b.children[col].dataset.val ?? b.children[col].textContent;
        const na = parseFloat(va), nb = parseFloat(vb);
        const cmp = (!isNaN(na) && !isNaN(nb))
          ? na - nb
          : va.localeCompare(vb, 'es', {{ sensitivity: 'base' }});
        return sortAsc ? cmp : -cmp;
      }})
      .forEach(tr => tbody.appendChild(tr));
  }}

  input.addEventListener('input', actualizar);
  ths.forEach((th, i) => th.addEventListener('click', () => ordenar(i)));
}})();
</script>
</body>
</html>
"""

FILA_TEMPLATE = (
    '<tr class="{clase}" data-busqueda="{busqueda}">'
    '<td class="dia-col"  data-val="{dia_num}">{dia}</td>'
    '<td class="mes-col"  data-val="{mes_num}">{mes}</td>'
    '<td class="anio-col" data-val="{anio}">{anio}</td>'
    '<td class="lugar-col">{lugar_html}</td>'
    '</tr>'
)


def generar_html(resultados: list[dict]) -> str:
    # Ordenar por año → mes_num → dia (cronologico)
    def sort_key(r):
        try:
            dia_n = int(r["dia"])
        except (ValueError, TypeError):
            dia_n = 99
        return (r["anio"], r["mes_num"], dia_n)

    resultados = sorted(resultados, key=sort_key)

    # Colorear grupos alternados por (año, mes_num)
    grupos: dict[tuple, int] = {}
    orden_grupo = 0
    for r in resultados:
        key = (r["anio"], r["mes_num"])
        if key not in grupos:
            grupos[key] = orden_grupo
            orden_grupo += 1

    num_carpetas = len({r["numero"] for r in resultados})
    filas = []

    for r in resultados:
        lugar = r["lugar"]
        lugar_html = (
            f'<span class="sin-datos">{lugar}</span>'
            if lugar in ("No especificado", "?")
            else lugar
        )
        clase = "grupo-par" if grupos[(r["anio"], r["mes_num"])] % 2 == 0 else "grupo-impar"
        busqueda = f"{r['dia']} {r['mes']} {r['anio']} {lugar} {r['carpeta']}".lower()

        try:
            dia_num = int(r["dia"])
        except (ValueError, TypeError):
            dia_num = 99

        filas.append("        " + FILA_TEMPLATE.format(
            clase=clase,
            busqueda=busqueda.replace('"', "&quot;"),
            dia=r["dia"],
            dia_num=dia_num,
            mes=r["mes"],
            mes_num=r["mes_num"],
            anio=r["anio"],
            lugar_html=lugar_html,
        ))

    return HTML_TEMPLATE.format(
        total=len(resultados),
        carpetas=num_carpetas,
        filas="\n".join(filas),
    )


# -- Punto de entrada ---------------------------------------------------------

if __name__ == "__main__":
    print("=== Extractor de Operativos Veterinarios ===\n")
    print(f"Directorio base: {BASE_DIR}\n")
    print("Procesando todas las imagenes (dia + mes + lugar desde el flyer, año desde mtime)...\n")

    resultados = procesar_carpetas()

    print("\nGenerando index.html...")
    html   = generar_html(resultados)
    salida = BASE_DIR / "index.html"
    with open(salida, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"OK Archivo generado: {salida}")
    print(f"   {len(resultados)} filas en la tabla\n")
    print("Muestra (primeros 10):")
    for r in resultados[:10]:
        print(f"  {r['dia']:>2} {r['mes']:<12} {r['anio']}  —  {r['lugar']}")
