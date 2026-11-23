# Behavioural-model comparison (cpg_rca vs PRINS vs LogSynthFSM)

Единый эксперимент сравнения **качества и скорости поведенческих моделей** трёх
методов на одном тестовом сервисе и одних и тех же train/test-исполнениях.

## Что сравнивается

| Метод | Тип | Что строит модель | Где лежит реализация |
|---|---|---|---|
| **cpg_rca** | code-based | FSM из CPG исходного кода (offline-пайплайн: trie → FLOW → FSM) | `builder.py`, `src/offline/` |
| **PRINS** | log-based | DFA из train-логов (MINT + hybrid determinization) | `baselines/PRINS/` (vendored replication package) |
| **LogSynthFSM** | log-based | FSM из train-логов через parsing + State Discovery (проверяется **только FSM-stage**, не полный pipeline синтеза/anomaly detection) | `baselines/LogSynthFSM-KDD-2026/` (vendored package) |

## Данные

- Тестовый сервис: `../../services/retryservice` (synthetic RetryService с
  четырьмя сценариями: `immediate_success`, `one_retry_success` — train;
  `two_retries_success`, `retry_exhausted` — test).
- CPG сервиса: `../../services/retryservice/export.dot` (Joern).
- Фикстура: `data/fixture_data/` — 80 исполнений (`logs.jsonl`,
  `executions.jsonl` со split и ground truth `expected_valid`) + скрытый
  call-site oracle (`oracle_events.jsonl`). Генератор:
  `python services/retryservice/retry_service.py --repetitions 20` (детерминирован).
- Независимо размеченные invalid-трассы: `invalid_traces.json` — разметка
  выведена из семантики исходника `retry_service.py`, **не** из какой-либо
  проверяемой FSM. `retry budget exhausted` (ERROR) — валидный терминальный
  сценарий, а не структурная недопустимость.

## Протокол

- Одинаковые train/test-исполнения для всех методов; train-данные используются
  только для обучения (PRINS, LogSynthFSM); test и скрытая разметка при
  построении моделей недоступны.
- Общий алфавит наблюдаемых событий: все сообщения канонизируются одной
  функцией `normalize_message` (числа/hex/uuid → `<*>`).
- Общая семантика **полного acceptance**: трасса принята iff модель потребляет
  *все* события от начального состояния и заканчивает в accepting/terminal;
  пропуск сообщений запрещён. Для LogSynthFSM (у состояний нет явных флагов)
  стартовые = состояния без входящих переходов, терминальные = без исходящих.
- Внутренние состояния моделей друг с другом не сравниваются.
- cpg_rca оценивается как **полный пайплайн**: offline-FSM + нормализация
  событий из online-стада (`src/online/trie_matcher.normalize_text`) перед
  матчингом — та же политика, что использует runtime-матчер. Без неё событие
  `temporary failure, retrying` не матчится с шаблоном `temporary failure
  retrying <*>`, и recall cpg_rca занижается до 0.25 — это артефакт слоя
  матчинга, а не качество модели.
- Entrypoint (`RetryService.execute`) разрешается по имени в CPG, Joern-id не
  зашиты.

## Метрики

`metrics.py` (покрыто unit-тестами, `tests/test_comparison_*.py` в корне репо):

- **Recall_valid** — доля полностью принятых допустимых трасс;
- **Specificity_invalid** — доля отклонённых независимо размеченных invalid-трасс,
  с разбиением на две группы: `invalid_known_alphabet` (все события трасс есть
  в алфавите модели — отказ здесь означает реальное понимание структуры) и
  `invalid_unknown_event` (в трассе есть событие вне алфавита модели — отказ
  тривиален и ничего не говорит об избирательности);
- **Balanced Accuracy** = (R+S)/2 (блок `combined`; N/A, если нет invalid-трасс);
- Recall_valid отдельно для **known** (сценарии из train) и **heldout** (только test);
- coverage событий, время построения, время проверки трасс;
- **log-to-code accuracy** для cpg_rca против скрытого call-site oracle
  (для методов без code-level выхода — N/A).

## Запуск

```bash
# из корня репозитория
python eval/comparison/run.py                 # использует кэш PRINS, если есть
python eval/comparison/run.py --retrain-prins # пересобрать модель PRINS (MINT)
python eval/comparison/run.py --out <dir>     # другой каталог отчёта
python -m pytest tests/ -q                    # unit-тесты метрик и адаптеров
```

Требования: Python из `.venv`, JDK (для MINT внутри PRINS). LogSynthFSM
дополнительно требует LLM-провайдера (`OPENAI_API_KEY` или ollama на
`localhost:11434`); без него метод честно помечается `blocked` — заглушек нет.

PRINS обучается в изолированном подпроцессе (`prins_train.py`), чтобы избежать
коллизии имён пакета `src` между репозиторием и vendored PRINS. Сообщения
кодируются стабильными токенами `E0000…` (MINT требует идентификаторы без
пробелов) — это входная кодировка, сам PRINS/MINT не изменяются.

## Выходы

В каталоге отчёта (по умолчанию `reports/full/`):

- `per_execution.csv` — метод, case_id, ground truth, решение, число
  обработанных событий, причина отказа, время проверки;
- `summary.csv` — метрики с числителями/знаменателями по блокам
  (`combined`, `valid_all`, `valid_known`, `valid_heldout`,
  `valid_test_split`, `invalid`) + блок `speed`;
- `config.json` — версии Python/Java, параметры методов, split, seeds,
  разрешённый entrypoint, статусы и причины blocked/error;
- `cpg_rca/` — `trie.json`, `fsm.json` (построенная модель);
- `PRINS/` — `prins_retryservice.pkl` (+ `.meta.json` с временем обучения),
  `vocab.json` (сообщение → id), `prins_work/` (входы/выходы MINT);
- `LogSynthFSM/` — `logsynthfsm_work/` (распарсенные события, FSM-CSV),
  когда провайдер доступен.



## Текущие результаты (80 valid + 6 invalid трасс)

| Метод | Recall all / known / heldout | Specificity | BA | Coverage | Build, с |
|---|---|---|---|---|---|
| cpg_rca | 1.00 / 1.00 / 1.00 | 0.833 (5/6) | 0.917 | 1.00 | ~9.1 |
| PRINS (k=2, u=1; 5 состояний) | 0.75 / 1.00 / 0.50 | 1.00 (6/6) | 0.875 | 0.833 | ~0.9 |
| LogSynthFSM (FSM-stage; 5 состояний, ollama llama3.2:3b) | 0.75 / 1.00 / 0.50 | 1.00 (6/6) | 0.875 | 0.833 | ~82 |







Интерпретация: модель из кода (cpg_rca) обобщает на отложенные сценарии
(heldout recall 1.00), т.к. retry-цикл извлечён из CFG целиком; обе
log-based модели точны на известном (1.00) и строго отвергают invalid
(6/6), но отсекают сценарий `retry_exhausted` (событие
`retry budget exhausted` вне их train-алфавита — unknown_event), отсюда
heldout recall 0.50 и BA 0.875 у обеих.

Про избирательность на отрицательных трассах: все 6 invalid-трасс в этой
фикстуре собраны только из событий train-алфавита (ни одна не содержит
`retry budget exhausted`), поэтому для всех трёх методов блок
`invalid_known_alphabet` совпадает с полным `invalid` (specificity cpg_rca
0.833 = 5/6 — настоящая избирательность, единственный пропуск —
`invalid-finish-without-attempt` из-за овер-аппроксимации CFG), а блок
`invalid_unknown_event` пуст (N/A). Тривиальные отказания по неизвестному
событию здесь проявляются не в specificity, а в recall: именно так
log-based модели теряют валидные heldout-трассы `retry_exhausted`.
Единственный FP cpg_rca (`invalid-finish-without-attempt`) — овер-
аппроксимация CFG: нулевая итерация цикла структурно разрешена, а её
невозможность — data-flow факт (непустой `range`), который FSM без value
analysis не видит.
Log-to-code accuracy cpg_rca = 1.0 (520/520 oracle-событий сопоставлены
правильному call-site).

Скорость проверки (блок `speed`): cpg_rca ~3.5e4 трасс/с, PRINS ~4e5,
LogSynthFSM ~2e4. Построение: cpg_rca ~9 c (парсинг CPG), PRINS ~1 c (MINT),
LogSynthFSM ~82 c (LLM State Discovery, 3b-модель локально в Docker).
