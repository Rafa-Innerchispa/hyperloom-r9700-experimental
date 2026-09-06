# Hyperloom R9700: evidencia por peticion y verificacion independiente

Fecha: 6 de septiembre de 2026. Tarea: `ops_d534d4fc606c`.
Rama: `chatgpt/hyperloom-r9700-independent-qa-20260906`.
Base anterior: `b854ed85f2489f62d34c261f89d8962afe7e5d2b`.
Commit de codigo creado ANTES de medir: `1aa389560` (prefijo Git).

## Resultado

**Suite delimitada Hyperloom: 386 PASS, 0 FAIL, 0 SKIP. Ejecucion real por proxy local: KEEP. Recalculo offline de 36 muestras: PASS. Aceptacion del worktree/SHA fisico AMD: pendiente.**

La suite final incorpora las 328 pruebas anteriores y 58 adicionales. Las ejecuciones parciales de 65 y 383 estan incluidas: no se suman al total. La advertencia preexistente `Unknown config option: asyncio_mode` permanece visible.

No se editaron archivos del control plane de Codex, no se reinicio vLLM, no se cambiaron permisos, no se llamaron modelos externos y no se modificaron main ni la rama original `9aa62d0654396b3f8246d2b79206e1406459cbc0`.

## Que cambia

El runner conserva las seis muestras individuales de cada ronda: indice de peticion, tiempo, tokens de entrada/salida, longitud de respuesta y resultado de la validacion. Los indices se vinculan a los futures de cada peticion y no al orden en que terminan. Con tres rondas por brazo se conservan 36 muestras medidas; los calentamientos no se incluyen.

El runner tambien registra SHA-256 de su archivo fuente al principio y al final, y la hora final del experimento. Si el archivo cambia durante la ejecucion, el run falla. Esta es una huella del archivo del runner, NO una firma, una atestacion de hardware o una huella de todas las dependencias.

El nuevo `scripts/r9700_evidence_audit.py` no importa el runner ni ejecuta modelos o red. Recalcula desde las muestras: tokens totales, tasas tokens/tiempo, medias, p95 por ronda mediante cuantiles inclusivos, agregaciones y KEEP/REJECT. Comprueba conteos, indices, tipos, rangos, cocientes y limites fijados en el auditor. Rechaza JSON con claves duplicadas, NaN/Infinity literales, archivos superiores a 1 MiB y huellas distintas de las esperadas cuando el llamante las proporciona.

El auditor separa expresamente:
- `metric_replay_verified`: consistencia de los datos y las cuentas.
- `physical_execution_verified`: siempre false en este auditor; requiere verificacion independiente del nodo.
- `global_fabric_verified`: false; no certifica toda la plataforma.
- `authentication_verified`: false; una huella almacenada junto al archivo no autentica su origen.

Los informes antiguos sin muestras no se reescribieron ni se presentan como si permitieran recalcular p95 desde latencias originales. El auditor los clasifica como no compatibles con esta verificacion por muestra.

## Nueva ejecucion real

Entrada: `scripts/r9700_upstream_local_proxy_e2e.py` mediante `peer_python_runtime`, nodo primary, proxy local existente, modelo `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ`.

| Medida | Resultado |
|---|---:|
| Mediana output tok/s baseline | 20.198822241208685 |
| Mediana output tok/s candidato | 36.3625039836844 |
| Ganancia de throughput | 80.02289217387799% |
| Cociente de medianas p95 | 1.0515054664534904 |
| Limite del cociente | 1.25 |
| Peticiones medidas validas | 36 |
| Fallos medidos | 0 |
| Concurrencia elegida por Qwen | 2, dentro de [1, 2] |
| Veredicto | KEEP |
| Estado del run / salida | completed / 0 |

La eleccion usa solamente las metricas baseline. Hubo un reintento acotado de formato, no una eleccion impuesta. El campo plan sigue vacio y no se inventa una explicacion del modelo.

[Informe con muestras originales](hyperloom_r9700_upstream_autonomous_e2e_20260906T170726136772Z.json).

SHA-256 de los bytes de ese JSON:
`0e8c626ee1c69a84f454e373aea5ae0edd17d25ccbfe4a909375a00ef8347034`

SHA-256 del archivo del runner observado antes y despues:
`311be5b6bf34c3baa0d77e49f43cb1edff565a386855dbab0d66510649d6374a`

[Salida original del auditor offline, preservada desde stdout MCP](r9700_sample_audit_live_20260906.json).

El primer audit CLI no recibio hashes esperados: sus campos expected_*_digest_checked son false. Despues, las pruebas `test_r9700_pinned_sample_evidence.py` cargaron ese archivo con los dos hashes fijados y confirmaron el recalculo. Tambien alteraron una latencia solamente en memoria y verificaron que la auditoria fallara. El JSON original no fue alterado.

La carga sigue siendo pequena: max_tokens=32 y tres rondas por brazo. Se conserva la mediana de los p95 por ronda, no un p95 global agrupado. No se demuestran significancia estadistica, ausencia de cache/efectos de orden ni rendimiento generalizable. El alcance sigue siendo `primary_orchestrated_existing_proxy_not_amd_worktree_acceptance`, con hardware_attested_by_runner=false. No es soporte oficial AMD ni paridad completa Instinct.

## Pruebas ejecutadas

[JUnit final Hyperloom](r9700_sample_audit_20260906_final.xml): 386 PASS. MCP command_run_id `230c4ad3ebc72efbd9791fec`.

La suite incluye contratos y seguridad del backend, loop autonomo, el flujo principal, las regresiones anteriores, 55 casos de captura/auditoria y tres casos del informe live fijado por hash. Las pruebas negativas usan fixtures o copias en memoria, no fallos provocados en servicios productivos.

Compileall scripts + src/kernelforge y git diff --check dieron salida 0.

Recalculo sin modelo ni red desde la raiz del repo:

```sh
python3 scripts/r9700_evidence_audit.py \
  docs/evidence/hyperloom_r9700_upstream_autonomous_e2e_20260906T170726136772Z.json \
  --expected-sha256 0e8c626ee1c69a84f454e373aea5ae0edd17d25ccbfe4a909375a00ef8347034 \
  --expected-runner-sha256 311be5b6bf34c3baa0d77e49f43cb1edff565a386855dbab0d66510649d6374a
```

## Verificacion de los avances de Codex

1. `get_mcp_fleet_status` consultado en esta sesion: MCP responde en ambos nodos y coinciden los seis archivos que compara ese monitor. No cubre todos los archivos de plataforma.
2. Se ejecutaron independientemente 50 pruebas en el worktree `codex/ops_05dc36bc0b51-durable-spine`: durable_coordination_spine, dev_swarm_repo_inference y a2a_bridge. Comando MCP `a738c626ebfd402ba8cb539d`. Son pruebas del worktree con fixtures; no equivalen a 50 operaciones live de NATS/Temporal.
3. El intento de ejecutar tests directamente en `/home/rlopez/inneros/inneros_core/platform` por peer fue rechazado `project_path_not_under_trusted_root`. No se eludio ese guard. Se distingue esa prueba no ejecutada de la suite permitida del worktree.
4. Codex reporto el fix `c50b36d0f20785a266c1a4b73145f941f95c73b4` para evidencia descartada en transiciones al mismo estado. Se verifico EN VIVO en nuestra tarea real, sin crear tareas-probe: update in_progress -> in_progress con evidencia y expected_revision=9 devolvio evidence_updated=true y revision=10. La lectura posterior de list_ops_tasks confirmo evidence no vacia, evidence_history y last_evidence_at. Se mantuvo owner=chatgpt. Este chequeo si demuestra persistencia en ese camino.
5. La respuesta del update devolvio event_id=null. No se afirma, a partir de esa respuesta, que se verificara la entrega de ese evento a NATS/OTel.

## Pendientes de aceptacion

La consulta peer del proyecto AMD `hyperloom-r9700-amd-live-verify` sigue mostrando ausente `scripts/r9700_upstream_agent_e2e.py`. El schema expuesto de project_runtime_bootstrap no permite ref/SHA. Se solicito a Codex el contrato invocable, no se reactivaron retries cancelados y no se confundio una cola vacia con un fallo del scheduler.

Falta certificar: `repo/ref/SHA esperado -> materializacion AMD -> SHA observado -> ejecucion dentro de ese worktree -> evidencia -> verificacion -> estado terminal`. Los avances de memoria/curator y la entrega durable completa siguen bajo sus tareas de plataforma, no cerradas por este informe.

## Borrador de publicacion, sin enviar

> Nuestro experimento Hyperloom con Qwen local ya conserva las metricas de cada peticion y dispone de un auditor que recalcula rendimiento, latencias y decision sin volver a llamar al modelo. La suite delimitada pasa 386 pruebas. En la nueva ejecucion orquestada desde Intel mediante el proxy local medimos 20,20 a 36,36 tokens de salida por segundo, con las restricciones de latencia configuradas satisfechas. Son resultados de una carga pequena y especifica. La comprobacion del commit y worktree fisicos en AMD sigue pendiente. El port R9700 es experimental y no representa soporte oficial de AMD.
