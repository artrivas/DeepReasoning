# DeepReasoning: razonamiento profundo mediante destilación cognitiva

Este repositorio contiene el proyecto **DeepReasoning**, cuyo objetivo es entrenar y evaluar un modelo de lenguaje compacto para resolver problemas matemáticos con razonamiento explícito, reflexión y respuesta final estructurada. El sistema sigue la idea de *cognitive distillation*: en vez de modificar la arquitectura del modelo, se le enseña mediante ajuste fino supervisado a imitar trazas de razonamiento con etiquetas semánticas.

El formato objetivo de las respuestas es:

```text
<thinking>
Razonamiento paso a paso.
</thinking>
<reflection>
Revisión o autocorrección del razonamiento.
</reflection>
<answer>
Respuesta final.
</answer>
```

La versión final del proyecto se enfoca en **Qwen/Qwen2.5-1.5B-Instruct**. Inicialmente se trabajó también con Qwen2.5-3B-Instruct, pero el modelo de 3B resultó menos diagnóstico para este dataset: su modelo base ya era suficientemente fuerte como para que la mejora por destilación fuera más difícil de aislar. Por eso, el análisis final usa el modelo de 1.5B, donde el efecto del ajuste fino es más visible.

## Artefactos principales

- [`qwen15b_presentation_executed.ipynb`](qwen15b_presentation_executed.ipynb): notebook ejecutado para presentación. Recalcula métricas desde predicciones crudas, genera gráficos y ejecuta una inferencia corta con el adaptador.
- [`qwen15b_presentation.ipynb`](qwen15b_presentation.ipynb): versión reproducible del notebook de presentación.
- [`report/main.pdf`](report/main.pdf): reporte final en formato académico.
- [`report/main.tex`](report/main.tex): fuente LaTeX del reporte.
- [`artifacts_qwen15b/`](artifacts_qwen15b/): resultados, predicciones, resúmenes, curvas y logs de la ablación con Qwen2.5-1.5B.
- [`deepreasoning.ipynb`](deepreasoning.ipynb): notebook base original del proyecto.
- [`FINAL_ABLATION_REPORT.md`](FINAL_ABLATION_REPORT.md): reporte de la ablación inicial con el modelo de 3B.

## Descripción del proyecto

El proyecto implementa un sistema de razonamiento profundo en tres fases:

1. **Destilación de datos**: se construye un dataset supervisado de problemas matemáticos tipo GSM8K con trazas de razonamiento estructuradas en `<thinking>`, `<reflection>` y `<answer>`.
2. **Ajuste fino con SFT + QLoRA**: se carga un modelo base eficiente y se entrenan únicamente adaptadores LoRA sobre pesos cuantizados en 4 bits.
3. **Escalado en inferencia**: se implementa la ruta para *self-consistency* o votación mayoritaria, donde se generan varias rutas de razonamiento para un mismo problema y se escoge la respuesta por consenso. Por costo de GPU, la confirmación final reportada usa inferencia greedy pareada (`N=1`), reservando `N>1` para candidatos prometedores.

El dominio elegido es razonamiento matemático estilo **GSM8K**, porque permite evaluar respuestas finales de forma determinista y al mismo tiempo exige razonamiento multi-paso.

## Modelo y entrenamiento

La configuración principal del experimento final fue:

| Componente | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Método | SFT con QLoRA |
| Cuantización | 4-bit, cómputo `bfloat16` |
| Longitud máxima | 1024 tokens |
| Batch físico | 8 |
| Acumulación de gradiente | 4 |
| Batch efectivo | 32 |
| Épocas | 3, con monitoreo de early stopping |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Módulos LoRA | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Se entrenaron y compararon tres configuraciones principales:

| Configuración | LoRA `r` | LoRA `alpha` | Learning rate |
|---|---:|---:|---:|
| `qwen15_len1024_r8_a16_lr5e5_e3_es` | 8 | 16 | `5e-5` |
| `qwen15_len1024_r8_a16_lr2e5_e3_es` | 8 | 16 | `2e-5` |
| `qwen15_len1024_r16_a32_lr5e5_e3_es` | 16 | 32 | `5e-5` |

También se preservaron intentos con batch físico mayor como registros de OOM. La configuración estable final mantuvo batch físico 8 con acumulación de gradiente.

## Evaluación

Se usaron dos formas de evaluación, porque miden cosas distintas:

- **Strict tagged answer accuracy**: exige que la respuesta final esté en el formato correcto con etiquetas. Mide obediencia al protocolo del proyecto.
- **Loose numeric answer accuracy**: extrae la respuesta numérica aunque el formato no sea perfecto. Mide mejor la capacidad matemática cruda.

Además se reportan:

- tasa de formato válido;
- tasa de reflexión;
- pérdida de validación;
- latencia media por ejemplo;
- comparación pareada contra el modelo base;
- intervalos bootstrap para la diferencia de accuracy.

## Resultados principales

En el *fast screen* de 30 problemas, la mejor configuración fue:

```text
max_len = 1024
LoRA r = 8
LoRA alpha = 16
learning_rate = 5e-5
```

Resultados de la pantalla rápida:

| Configuración | Eval loss | Accuracy 30 | Formato válido | Reflexión |
|---|---:|---:|---:|---:|
| `r=8, lr=5e-5` | 0.402 | 63.3% | 90.0% | 90.0% |
| `r=8, lr=2e-5` | 0.456 | 56.7% | 80.0% | 80.0% |
| `r=16, lr=5e-5` | 0.383 | 53.3% | 80.0% | 83.3% |

Un hallazgo importante es que menor pérdida de validación no implicó mejor respuesta final: el modelo con `r=16` tuvo la mejor `eval_loss`, pero peor accuracy que `r=8, lr=5e-5`.

En la confirmación de 100 problemas para el mejor adaptador:

| Métrica | Base | Adaptador | Diferencia |
|---|---:|---:|---:|
| Strict tagged answer accuracy | 0.0% | 48.0% | +48.0 |
| Loose numeric answer accuracy | 55.0% | 49.0% | -6.0 |
| Formato válido | 0.0% | 75.0% | +75.0 |
| Reflexión | 0.0% | 75.0% | +75.0 |
| Latencia media | 4.19 s | 15.69 s | +11.49 s |

La conclusión honesta es que el adaptador aprende muy bien el **protocolo de razonamiento estructurado**, pero no supera de forma confiable al modelo base en accuracy numérica suelta. Es decir, la destilación mejora la forma, la trazabilidad y la facilidad de evaluación, pero todavía no garantiza mejor razonamiento matemático final.

## Cómo reproducir

### Ejecutar la ablación principal de Qwen2.5-1.5B

```bash
chmod +x run_qwen15b_screen.sh
./run_qwen15b_screen.sh
```

### Confirmar el mejor adaptador en 100 problemas

```bash
python evaluate_qwen15_confirm.py
```

### Reconstruir el notebook de presentación

```bash
python build_qwen15_presentation.py
jupyter nbconvert \
  --to notebook \
  --execute qwen15b_presentation.ipynb \
  --output qwen15b_presentation_executed.ipynb
```

### Ejecutar la ablación original de 3B

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

### Ejecutar el mejor experimento original

```bash
chmod +x run_best.sh
./run_best.sh
```

## Estructura del repositorio

```text
.
├── qwen15b_presentation.ipynb              # Notebook reproducible de presentación
├── qwen15b_presentation_executed.ipynb     # Notebook ejecutado con gráficos e inferencia
├── build_qwen15_presentation.py            # Generador del notebook de presentación
├── run_qwen15b_screen.sh                   # Runner de la ablación Qwen2.5-1.5B
├── evaluate_qwen15_confirm.py              # Confirmación de 100 problemas
├── artifacts_qwen15b/                      # Resultados y evidencia cruda Qwen2.5-1.5B
├── report/                                 # Reporte final en LaTeX/PDF
├── deepreasoning.ipynb                     # Notebook base original
├── run_ablation.sh                         # Ablación original del modelo 3B
├── run_best.sh                             # Mejor corrida original
└── FINAL_ABLATION_REPORT.md                # Reporte de la ablación original
```

## Conclusión

El proyecto demuestra que SFT + QLoRA puede transformar un modelo pequeño en un sistema que produce razonamiento explícito y verificable. La mejora más fuerte aparece en métricas de estructura: formato válido, presencia de reflexión y respuestas parseables dentro de etiquetas. Sin embargo, el resultado también muestra una limitación clave: producir una cadena de pensamiento ordenada no garantiza que la respuesta matemática final sea correcta.

Por eso, el mejor uso de este sistema no es afirmar que el adaptador reemplaza al modelo base en capacidad matemática general, sino que ofrece una base más controlable para evaluación, análisis de errores y futuros métodos de *test-time compute* como self-consistency.

## Trabajo futuro

- Evaluar con múltiples semillas.
- Aumentar y filtrar mejor el dataset destilado.
- Usar un LLM externo como juez para medir rigor lógico y autocorrección.
- Ejecutar self-consistency `N>1` solo en los mejores adaptadores o problemas difíciles.
- Comparar contra más modelos compactos.
- Separar métricas de obediencia al formato y métricas de razonamiento matemático puro.
