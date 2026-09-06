# Hyperloom R9700: verificacion independiente del 6 de septiembre de 2026

## Resultado y alcance

**Validacion y suite delimitada: PASS, 320 pruebas. Dos ejecuciones reales del agente por el proxy local: KEEP. Ejecucion del ref exacto dentro del worktree AMD: BLOCKED. Execution Fabric global: no certificado.**

Repositorio: `Rafa-Innerchispa/hyperloom-r9700-experimental`.
Rama exclusiva: `chatgpt/hyperloom-r9700-independent-qa-20260906`.
Base original preservada: `9aa62d0654396b3f8246d2b79206e1406459cbc0`.
Primer commit del parche de validacion: `d617e2d8944072ab3a82ac096770b4cc78d87a13`.
El commit que contiene esta revision del informe fija tambien los scripts de ejecucion, las pruebas y los dos JSON de evidencia enlazados abajo. Las ejecuciones live corresponden al codigo corregido de la rama QA, NO al commit original sin modificar.
Tarea: `ops_d534d4fc606c`; correlacion: `hyperloom-independent-qa-20260906`.

No se tocaron los archivos del control plane que repara Codex, la rama original, main, modelos residentes, permisos de host o servicios productivos. No se reinicio vLLM, no se lanzaron proveedores externos y no se uso force-push.

## Fallos reproducidos y corregidos

| Fallo | Reproduccion previa | Correccion |
|---|---|---|
| Respuestas vacias, mal formadas o ajenas a la consigna | HTTP exitoso se contabilizaba como respuesta valida | Exigir estructura, texto no vacio y la palabra AMD antes de contabilizar tokens |
| Conteos de tokens cero, negativos, booleanos o fraccionarios | Aceptacion o conversion silenciosa | Enteros estrictos, completion_tokens positivo y prompt_tokens no negativo |
| Throughput infinito o latencia cero, negativa o infinita | Podia producir KEEP | Metricas y cocientes finitos y positivos |
| Conteos incompletos o inconsistentes | Se aceptaba el resumen de medianas | Exigir 3 rondas, 18 peticiones y conteos coherentes por brazo |
| Evidencia ausente | KeyError | REJECT estructurado |
| Candidato ambiguo o con instrucciones extra | Se tomaba solo la primera asignacion coincidente | Una unica asignacion CONCURRENCY de valor 1 o 2; comentarios permitidos; archivo acotado |
| Modelo devuelve un candidato con formato invalido | La ejecucion se interrumpia sin oportunidad acotada de corregirlo | Un unico reintento de formato, registrado, sin imponer un valor ni mostrar resultados del candidato |
| Afirmacion de hardware sin comprobacion | Un texto fijo podia confundirse con atestacion | Registrar host orquestador, alcance y hardware_attested_by_runner=false |

Antes del parche, las 34 nuevas pruebas adversariales dieron **21 fallos y 13 aprobaciones**. Despues, las mismas 34 dieron **34 aprobaciones**. La comprobacion de la palabra AMD solo valida esta consigna de benchmark; no certifica la correccion general del modelo ni kernels GPU. El candidato se interpreta como datos: su contenido no se ejecuta.

## Ejecucion real del agente

Entrada: `scripts/r9700_upstream_local_proxy_e2e.py`, ejecutada mediante `peer_python_runtime` en **primary**. Utiliza el proxy loopback existente en el puerto 18000 y el modelo residente `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`. No cambia configuracion del host.

Flujo real: `KernelForge make_agent_fn -> local-openai registrado -> Qwen -> write_file -> benchmark -> gate`.

| Informe UTC | Baseline mediana output tok/s | Candidato mediana output tok/s | Ganancia | Cociente mediana p95 | Decision |
|---|---:|---:|---:|---:|---|
| 2026-09-06 15:57:15 | 20.0399099203 | 36.3732669609 | 81.504144% | 1.0558529432 | KEEP |
| 2026-09-06 15:58:32 | 20.1006912729 | 36.1251249626 | 79.720809% | 1.0447417366 | KEEP |

En ambos registros hay tres rondas baseline y tres candidate, seis peticiones por ronda, 18 respuestas validas por brazo y cero fallos medidos. Son 72 peticiones medidas entre ambos experimentos, sin incluir calentamientos ni llamadas del agente.

Qwen eligio concurrencia 2 dentro de [1, 2], despues de recibir solamente las metricas baseline. En ambos casos necesito **un reintento de formato**: la primera escritura no cumplio el contrato de una unica asignacion; la segunda si. Los JSON conservan validation_attempts, format_repair_used y el progreso write_file. El campo plan quedo vacio; no se presenta una explicacion del modelo que no exista.

Evidencia original de las mediciones:
- [Primer experimento](hyperloom_r9700_upstream_autonomous_e2e_20260906T155715Z.json).
- [Segundo experimento](hyperloom_r9700_upstream_autonomous_e2e_20260906T155832Z.json).

Los diez casos de `scripts/tests/test_r9700_recorded_proxy_evidence.py` recalculan ambos informes de forma independiente: conteos, tasas tokens/tiempo, todas las agregaciones publicadas, limites del gate, decision, contexto del modelo y limites de atestacion. Son verificaciones offline de resultados guardados, no diez benchmarks nuevos.

## Pruebas automatizadas finales

**320 PASS, sin fallos ni pruebas omitidas en la suite delimitada ejecutada.** Incluye scripts, backend local-openai, integracion upstream, bucle autonomo, contratos, sandbox, rutas y registro de proveedores. Las ejecuciones parciales de 34, 305 y otras estan incluidas o preceden esta suite: no se suman para inflar el total.

[JUnit final, generado por pytest](r9700_independent_qa_20260906_final.xml).
La advertencia del entorno compartido `Unknown config option: asyncio_mode` sigue presente; no se oculto.
MCP command_run_id final: `c3b63015cb3dd6aa8c9e947a`.
Compileall de scripts y kernelforge: salida 0, `8ad5ca278536764bbca897a6`.

El control plane reutiliza command_run_id al repetir el mismo comando. Estos IDs deben acompanarse de revision, timestamp y salida; no equivalen a un identificador unico de cada intento.

```sh
python3 -m pytest scripts/tests \
  src/kernelforge/tests/test_local_openai_backend.py \
  src/kernelforge/tests/test_local_openai_config_integration.py \
  src/kernelforge/tests/test_local_openai_upstream_integration.py \
  src/kernelforge/tests/test_r9700_autonomous_loop.py \
  src/kernelforge/tests/test_r9700_upstream_autonomous_e2e.py \
  src/kernelforge/tests/test_agent_run_spec_contract.py \
  src/kernelforge/tests/test_agent_sandbox_policy.py \
  src/kernelforge/tests/test_workspace_policy.py \
  src/kernelforge/tests/test_path_ownership.py \
  src/kernelforge/tests/test_provider_registry.py \
  -q --tb=short --junitxml=docs/evidence/r9700_independent_qa_20260906_final.xml
python3 -m compileall -q scripts src/kernelforge
git diff --check
```

## Lo que estas pruebas NO demuestran

El alcance explicito de ambas ejecuciones es `primary_orchestrated_existing_proxy_not_amd_worktree_acceptance`, con `hardware_attested_by_runner=false`. Son inferencias reales a traves del proxy local, pero no prueban que el SHA solicitado este materializado y ejecutado fisicamente dentro del worktree AMD. Tampoco certifican reparaciones de task binding, approvals, stale recovery ni completed con evidencia en el Execution Fabric.

La comprobacion peer posterior sigue devolviendo `exists=false` para `scripts/r9700_upstream_agent_e2e.py` en el proyecto AMD `hyperloom-r9700-amd-live-verify`. El bootstrap expuesto no admite ref/SHA. No se usaron rutas alternativas para eludir denegaciones de permisos.

Continua pendiente: `repo/ref/SHA esperado -> materializar en AMD -> SHA observado igual -> ejecucion en ese nodo -> heartbeat -> evidencia -> verificacion -> estado terminal`.

Se solicito checkpoint a Codex y se envio el resultado 320 PASS + KEEP al hilo de su P0. Al ultimo chequeo no habia una respuesta nueva en los mensajes enviados de Codex consultados. Esto no demuestra que su otra instancia este detenida. Sus tareas `ops_bf21134585d4` y `ops_05dc36bc0b51` no fueron cerradas por este trabajo.

## Limites de rendimiento y publicacion

El gate conserva ganancia minima 10% y cociente maximo de latencia 1.25. Utiliza **mediana del throughput de tres rondas** y **mediana de los p95 de tres rondas**, no un p95 global agrupado. La carga es pequena, con max_tokens=32. No se ha demostrado significancia estadistica, ausencia de efectos de cache o de orden, ni ganancia generalizable a produccion.

El resultado corresponde a concurrencia y optimizacion neutral respecto de arquitectura. No es tuning de kernels gfx1201 ni paridad Instinct. Profiling profundo RDNA4, TraceLens/Magpie adaptados y runners CDNA completos siguen fuera del alcance. Shell permanece deshabilitado y no se habilitaron rutas MI300/gfx942/gfx950.

### Borrador de actualizacion, no publicado

> Nuestro experimento Hyperloom con Qwen local ya completa el ciclo de medir, elegir un candidato, volver a medir y aceptar o rechazar con reglas fijas. Una auditoria independiente encontro y corrigio casos de validacion que podian producir resultados enganosos. La suite delimitada pasa 320 pruebas. Dos ejecuciones reales, orquestadas desde nuestro nodo Intel mediante el proxy local, registraron ganancias de throughput de 81.50% y 79.72%, dentro del limite configurado de latencia. Son mediciones de una carga pequena y especifica, no una promesa de rendimiento general. La verificacion del worktree y SHA exactos en AMD sigue pendiente. Este port para Radeon AI PRO R9700 es experimental: no representa soporte oficial de AMD.

No se publico en redes ni se promociono el parche a main. Esta evidencia permite informar del avance delimitado, pero no afirmar que todo el Fabric o el port completo esten terminados.
