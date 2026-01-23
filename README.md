# Система экспертной оценки недвижимости (MVP)

FastAPI-приложение, которое принимает JSON объекта недвижимости, подставляет его в PROMPT, обращается к LLM (cloud или заглушка), валидирует ответ и показывает структурированный отчёт. История запросов сохраняется в SQLite (пока жив контейнер, либо в persistent volume).

## Архитектура
- `FastAPI` + `Jinja2` + Bootstrap (CDN).
- Провайдеры LLM: `CloudProvider` (HTTP API) и `LocalStubProvider`. Переключение через `AI_MODE=cloud|stub`.
- PROMPT-файл: `prompts/valuation_prompt.md` с плейсхолдером `{{PROPERTY_JSON}}`.
- История в `data/history.db` (создаётся автоматически).

## Переменные окружения
- `PORT` — обязательный для Amvera, порт прослушивания (по умолчанию 8000).
- `AI_MODE` — `cloud` (по умолчанию) или `stub`.
- `CLOUD_API_URL` — endpoint LLM (chat/completions). Для GPT: `https://api.openai.com/v1/chat/completions`.
- `CLOUD_API_KEY` — ключ для LLM (`sk-...` для OpenAI).
- `CLOUD_MODEL` — имя модели (по умолчанию `gpt-4o-mini`).
- `CLOUD_TIMEOUT` — таймаут запроса в секундах (по умолчанию 20).

Скопируйте `.env.example` в `.env` и заполните ключи.

## Локальный запуск
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export $(cat .env | xargs)  # или вручную
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```
Откройте `http://localhost:8000/` для формы, `http://localhost:8000/history` для истории, `http://localhost:8000/health` для проверки.

## Docker
```bash
docker build -t valuation-app .
docker run --rm -p 8000:8000 --env-file .env valuation-app
```
Приложение слушает `0.0.0.0:$PORT`. История сохраняется в контейнере; подключите volume к `/app/data`, если нужен персистент.

## Деплой на Amvera
1. Соберите образ (либо через CI, либо `docker build`).
2. В Amvera укажите переменные окружения: `PORT`, `AI_MODE`, `CLOUD_API_URL`, `CLOUD_API_KEY`, `CLOUD_MODEL` (опционально).
3. Откройте порт `$PORT` (платформа прокинет его в контейнер).
4. Команда запуска по умолчанию в образе: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
5. Health-check: `GET /health` должен вернуть `{"status":"ok"}`.
6. Логи смотрите в UI Amvera или `docker logs <container>` (если локально).
7. Для сохранения истории добавьте volume к `/app/data` (иначе база живёт пока жив контейнер).

## Страницы и API
- `/` — форма ввода JSON, вывод отчёта, кнопка «Показать сырой ответ».
- `/history` — список последних запросов.
- `/history/{id}` — подробности записи.
- `POST /analyze` — принимает JSON тела (`{"property_data": {...}}`) или form-data `property_json`.
- `GET /api/history?limit=20` — JSON-история.
- `GET /health` — проверка готовности.

## Типовой сценарий демонстрации
1. Открыть `/`, вставить пример JSON (кнопка уже содержит шаблон).
2. Нажать «Оценить», дождаться отчёта: summary, recommendation, risk_score, диапазон цены, списки pros/cons.
3. Раскрыть «Показать сырой ответ», показать строгий JSON.
4. Перейти в `/history`, показать запись, кликнуть на id — открыть детали с входом/выходом.
5. Переключить `AI_MODE=stub` для оффлайн-демо — получим детерминированный ответ.

## Обработка ошибок
- Неверный JSON входа, отсутствие `type` или `location/address/city`, площадь ≤ 0 — понятное сообщение на UI и в API.
- Ошибки LLM: таймаут, HTTP-ошибка, некорректный формат — показывается сырой ответ + сообщение, запись сохраняется в истории.
