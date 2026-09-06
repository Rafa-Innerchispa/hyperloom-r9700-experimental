# Seguimiento independiente: estado de ejecucion y mediciones

Fecha: 6 de septiembre de 2026. Tarea: `ops_d534d4fc606c`.
Rama: `chatgpt/hyperloom-r9700-independent-qa-20260906`.
Base de esta continuacion: `6d4e45f72c2b29c448f1e684efb212ea0177d416`.

## Que se avanzo sin modificar el control plane

Se reprodujo un defecto adicional del runner: aunque el gate rechazaba mediciones invalidas, main imprimia `ok=true` y devolvia codigo 0. Por ejemplo, un brazo con 18 peticiones fallidas terminaba como una ejecucion exitosa. Ademas, una metrica infinita podia aparecer como `Infinity`, que no es JSON estandar.

Ahora el runner distingue dos resultados:

- Experimento valido: `ok=true`, `run_status=completed`, codigo 0. Su candidato puede tener veredicto KEEP o REJECT.
- Medicion o ejecucion invalida: `ok=false`, `run_status=failed`, codigo 2, sin veredicto de rendimiento. Conserva un informe con etapa y motivo del fallo.

Si falla la medicion baseline, no se llama al agente ni se mide el candidato. Los fallos de descubrimiento del modelo y seleccion del candidato tambien producen informes estructurados. Los numeros no finitos se representan como null, con politica explicita de serializacion y estado fallido. Las excepciones no copian mensajes arbitrarios al informe.

## Pruebas

Las ocho nuevas pruebas fallaron antes del cambio. Incluyen seis condiciones de fallo y dos controles de finalizacion valida; no representan ocho defectos independientes. Tras el parche, pasan junto con las dos pruebas previas de flujo main: 10 PASS.

Suite delimitada completa: **328 PASS**, sin fallos ni omisiones; una advertencia existente de configuracion pytest `asyncio_mode`. Son las 320 pruebas anteriores mas ocho nuevas, no un total acumulado de ejecuciones repetidas.

- JUnit generado por pytest: [r9700_run_status_20260906.xml](r9700_run_status_20260906.xml).
- Regresiones: `scripts/tests/test_r9700_run_failure_status.py`.
- MCP prueba negativa: `b04a90f2615a154f94e1a111`.
- MCP suite final: `173b43544c45f540f9c7dca3`.
- Compileall de scripts y kernelforge: codigo 0.

Las regresiones de errores usan fixtures sinteticos sin red, no fallos provocados en produccion.

## Repeticion real despues del parche

Se ejecuto una vez `scripts/r9700_upstream_local_proxy_e2e.py` mediante peer_python_runtime en primary, usando el proxy local existente y Qwen3-Coder-30B.

[JSON original de esta ejecucion](hyperloom_r9700_upstream_autonomous_e2e_20260906T165116889777Z.json).

| Medida | Resultado |
|---|---:|
| Mediana output tok/s baseline | 20.0111805785 |
| Mediana output tok/s candidato | 36.3368923375 |
| Ganancia | 81.5829515647% |
| Cociente de medianas p95 | 1.0508175078 |
| Rondas por brazo | 3 |
| Peticiones validas por brazo | 18 |
| Fallos medidos | 0 |
| Concurrencia elegida por el modelo | 2 |
| Veredicto | KEEP |
| Estado de ejecucion | completed |
| Codigo de salida | 0 |

La eleccion siguio siendo autonoma entre 1 y 2. Hubo un reintento de formato, registrado. El campo plan del agente sigue vacio; no se atribuye al modelo una explicacion inexistente.

La carga es pequena, con max_tokens=32. Las metricas son medianas de tres rondas, no un p95 global ni una garantia estadistica de produccion. Esta repeticion se realizo con la nueva revision QA, no con el SHA original 9aa62d065.

## Verificacion independiente del avance de Codex

El checkpoint `msg_b917e34bf443aec9` reporto el commit de plataforma `24ee4e00bdb52c8499dc2bd2bc587b9ece075134` para arreglar el enrutamiento de tareas de texto hacia vLLM local cuando Ollama no esta disponible.

La prueba independiente run_local_model(status_probe) devolvio el texto solicitado `RUTA LOCAL VERIFICADA 20260906`, runtime local_vllm, selected_node=amd, external_needed=false. Response id: `chatcmpl-a41883b2ef68648d`. Este resultado confirma esa ruta de inferencia, no la reparacion completa del Fabric.

## Limite que sigue abierto

La consulta fisica del proyecto `hyperloom-r9700-amd-live-verify` en AMD volvio a devolver exists=false para `scripts/r9700_upstream_agent_e2e.py`. El bootstrap MCP disponible sigue sin parametro ref/SHA. La prueba local tiene alcance `primary_orchestrated_existing_proxy_not_amd_worktree_acceptance` y hardware_attested_by_runner=false.

Por tanto sigue pendiente verificar materializacion y SHA en AMD antes de certificar el E2E de ese worktree. No se tocaron los archivos de plataforma que trabaja Codex, ni se reinicio vLLM, ni se ampliaron permisos, ni se cambio main o la rama original. No es soporte oficial AMD Hyperloom para R9700.
