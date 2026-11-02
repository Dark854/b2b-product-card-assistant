import json
import logging
import os
from html import escape

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
import httpx
from google import genai
from google.genai import errors
from google.genai import types
from dotenv import load_dotenv
from pydantic import ValidationError

from schemas import InputProductData, OutputProductCard
from prompts import SYSTEM_INSTRUCTION, build_user_prompt

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
logger = logging.getLogger(__name__)
client = (
    genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=5_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    if api_key
    else None
)

app = FastAPI(title="B2B AI Product Description Generator")


def create_local_draft(
    data: InputProductData, warning: str
) -> OutputProductCard:
    specs_text = "; ".join(
        f"{key}: {value}" for key, value in data.specs.items()
    )
    if not specs_text:
        specs_text = "характеристики во входных данных не указаны"

    description = " ".join(
        (
            f"{data.name} — производитель: {data.manufacturer}.",
            data.description[:300],
            f"Во входных данных указаны характеристики: {specs_text[:250]}.",
            "Это локальный черновик: сервис генерации сейчас недоступен.",
            "В тексте использованы только сведения, переданные в форме. "
            "Дополнительные технические свойства, совместимость и область "
            "применения не предполагаются.",
            "Характеристики приведены по исходным данным и не заменяют "
            "паспорт или техническую документацию изготовителя.",
            "Перед подбором прибора, закупкой или публикацией сверьте "
            "все параметры с паспортом и уточните сведения, важные для "
            "ваших условий эксплуатации, монтажа и подключения.",
            "Если нужного параметра нет во входных данных, считайте его "
            "неподтверждённым и запросите у поставщика.",
        )
    )
    while len(description) < 800:
        description += (
            " Этот черновик предназначен для проверки контент-менеджером "
            "и не является подтверждением характеристик."
        )
    description = description[:1200]

    title = f"{data.name} — {data.manufacturer}"[:70]
    search_queries = [
        f"{data.name} {data.manufacturer}",
        f"{data.name} характеристики",
        f"{data.manufacturer} {next(iter(data.specs.values()), data.name)}",
    ]
    structured_specs = dict(data.specs)
    if not structured_specs:
        structured_specs["Характеристики"] = "данных нет"

    commercial_block = (
        f"По предоставленному описанию: {data.description[:400]} "
        "Данных недостаточно, чтобы подтвердить совместимость или "
        "рекомендовать прибор для конкретной задачи. Перед подбором "
        "сверьте параметры с паспортом изготовителя."
    )

    return OutputProductCard(
        missing_fields_warning=warning,
        seo_title=title,
        seo_description=description,
        structured_specs=structured_specs,
        search_queries=search_queries,
        commercial_block=commercial_block,
    )


def generate_card_from_llm(data: InputProductData) -> OutputProductCard:
    if client is None:
        warning = (
            "Ключ Gemini не настроен. Показан локальный черновик, "
            "составленный только из введённых данных."
        )
        return create_local_draft(data, warning)

    prompt = build_user_prompt(data.model_dump())
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )
        raw_text = (response.text or "").strip()
        if raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()
            if raw_text[:4].casefold() == "json" and (
                len(raw_text) == 4 or raw_text[4].isspace()
            ):
                raw_text = raw_text[4:].lstrip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed_res = json.loads(raw_text.strip())
        if not isinstance(parsed_res, dict):
            raise ValueError("LLM response must be a JSON object")

        model_specs = parsed_res.get("structured_specs")
        if not isinstance(model_specs, dict):
            raise ValueError("structured_specs must be a JSON object")

        source_specs = {
            key.casefold(): (key, value) for key, value in data.specs.items()
        }
        structured_specs = dict(data.specs)
        for key, value in model_specs.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("structured_specs keys and values must be strings")
            source_spec = source_specs.get(key.casefold())
            if source_spec is None:
                structured_specs[key] = "данных нет"
            else:
                structured_specs[source_spec[0]] = source_spec[1]
        parsed_res["structured_specs"] = structured_specs

        return OutputProductCard(**parsed_res)
    except (errors.APIError, httpx.HTTPError) as exc:
        logger.warning(
            "Gemini request failed; returning a local draft (%s)",
            getattr(exc, "code", type(exc).__name__),
        )
        return create_local_draft(
            data,
            "Сервис ИИ недоступен или не ответил за 5 секунд. "
            "Показан локальный черновик только из введённых данных; "
            "проверьте его перед публикацией.",
        )
    except (
        AttributeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        logger.warning("Gemini returned an invalid card; using a local draft")
        return create_local_draft(
            data,
            "Сервис ИИ вернул некорректную карточку. Показан локальный "
            "черновик только из введённых данных; проверьте его перед публикацией.",
        )


@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>AI Карточка товара B2B</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; background: #f4f6f9; color: #333; }
            .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 180px; font-family: monospace; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { background: #0066cc; color: #fff; border: none; padding: 12px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #0052a3; }
            button:disabled { background: #6689ad; cursor: wait; }
            .progress { color: #334e68; margin-top: 12px; }
            pre { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 14px; }
            .warning { background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; border: 1px solid #ffeeba; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Генерация B2B карточки товара</h2>
            <form id="generate-form" action="/generate-ui" method="post">
                <label><b>Входной JSON товара:</b></label><br><br>
                <textarea name="json_input">{
  "name": "Манометр показывающий DM 02",
  "manufacturer": "Манотомь",
  "specs": {
    "диапазон измерений": "0–1,6 МПа",
    "класс точности": "1,5",
    "присоединение": "G1/2",
    "диаметр корпуса": "100 мм"
  },
  "description": "Прибор предназначен для измерения избыточного давления сухих, неагрессивных к медным сплавам газов и жидкостей."
}</textarea>
                <button id="generate-button" type="submit">Сгенерировать карточку</button>
                <p id="generation-status" class="progress" role="status" aria-live="polite" hidden>
                    Отправляем запрос модели...
                </p>
            </form>
        </div>
        <script>
            const form = document.getElementById("generate-form");
            const button = document.getElementById("generate-button");
            const status = document.getElementById("generation-status");
            form.addEventListener("submit", () => {
                button.disabled = true;
                button.textContent = "Генерируем...";
                status.hidden = false;
                status.textContent = "Запрос отправлен. Обычно генерация занимает несколько секунд.";
                window.setTimeout(() => {
                    status.textContent = "Если ИИ-сервис недоступен, будет показан локальный черновик.";
                }, 4000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate-ui", response_class=HTMLResponse)
def generate_ui(json_input: str = Form(...)):
    try:
        raw_json = json.loads(json_input)
        if not isinstance(raw_json, dict):
            raise ValueError("Входной JSON должен содержать объект товара.")
        input_data = InputProductData(**raw_json)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        message = escape(str(exc))
        return HTMLResponse(
            content=(
                "<h3>Ошибка во входных данных JSON:</h3>"
                f"<p>{message}</p><a href='/'>Назад</a>"
            ),
            status_code=400,
        )

    try:
        result = generate_card_from_llm(input_data)
    except HTTPException as exc:
        message = escape(str(exc.detail))
        return HTMLResponse(
            content=(
                "<h3>Не удалось сгенерировать карточку:</h3>"
                f"<p>{message}</p><a href='/'>Вернуться к форме</a>"
            ),
            status_code=exc.status_code,
            headers=exc.headers,
        )
    result_json = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    warning_html = ""
    if result.missing_fields_warning:
        warning_html = (
            "<div class='warning'><b>Внимание (неполные данные):</b> "
            f"{escape(result.missing_fields_warning)}</div>"
        )

    spec_rows = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"
        for key, value in result.structured_specs.items()
    )
    search_queries_html = "".join(
        f"<li>{escape(query)}</li>" for query in result.search_queries
    )
    copy_content = "\n".join(
        (
            f"SEO-заголовок\n{result.seo_title}",
            f"SEO-описание\n{result.seo_description}",
            "Характеристики",
            *(
                f"{key}: {value}"
                for key, value in result.structured_specs.items()
            ),
            "Поисковые запросы",
            *result.search_queries,
            f"Для менеджера\n{result.commercial_block}",
        )
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Результат генерации</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background: #f4f6f9; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 14px; }}
            .warning {{ background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; border: 1px solid #ffeeba; margin-bottom: 15px; }}
            a {{ text-decoration: none; color: #0066cc; font-weight: bold; }}
            .product-card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 22px; margin-top: 20px; }}
            .product-card h3 {{ margin-top: 0; color: #243b53; }}
            .product-card h4 {{ margin-bottom: 8px; color: #334e68; }}
            .product-card p {{ line-height: 1.6; white-space: pre-wrap; }}
            .specs {{ width: 100%; border-collapse: collapse; }}
            .specs th, .specs td {{ padding: 9px 12px; border-bottom: 1px solid #e6edf3; text-align: left; }}
            .specs th {{ width: 38%; color: #334e68; }}
            .copy-button {{ background: #0066cc; color: #fff; border: 0; border-radius: 4px; padding: 10px 16px; cursor: pointer; }}
            .copy-status {{ margin-left: 10px; color: #287d3c; }}
            details {{ margin-top: 24px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/">&larr; Вернуться к форме</a>
            <h2>Сгенерированная карточка товара</h2>
            {warning_html}
            <article class="product-card">
                <h3>{escape(result.seo_title)}</h3>
                <h4>SEO-описание</h4>
                <p>{escape(result.seo_description)}</p>
                <h4>Характеристики</h4>
                <table class="specs"><tbody>{spec_rows}</tbody></table>
                <h4>Поисковые запросы</h4>
                <ul>{search_queries_html}</ul>
                <h4>Для менеджера</h4>
                <p>{escape(result.commercial_block)}</p>
            </article>
            <p>
                <button class="copy-button" id="copy-card" type="button">Копировать карточку</button>
                <span class="copy-status" id="copy-status" role="status" aria-live="polite"></span>
            </p>
            <textarea id="copy-content" hidden>{escape(copy_content)}</textarea>
            <details>
                <summary>Показать JSON</summary>
                <pre>{escape(result_json)}</pre>
            </details>
        </div>
        <script>
            document.getElementById("copy-card").addEventListener("click", async () => {{
                const content = document.getElementById("copy-content").value;
                const status = document.getElementById("copy-status");
                try {{
                    await navigator.clipboard.writeText(content);
                    status.textContent = "Карточка скопирована.";
                }} catch {{
                    status.textContent = "Не удалось скопировать. Выделите текст вручную в JSON.";
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/v1/generate", response_model=OutputProductCard)
def api_generate_card(data: InputProductData):
    return generate_card_from_llm(data)