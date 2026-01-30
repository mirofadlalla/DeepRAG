import os
import logging
from huggingface_hub import InferenceClient
from zenml import step
from dotenv import load_dotenv
import mlflow
import time

load_dotenv(dotenv_path="E:\pyDS\Buliding Rag System\.env")

class LlamaInstructLLMGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.client = InferenceClient(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )
        self.model_name = model_name

        # System instructions
        self.system_prompt = """
أنت مساعد ذكي.
مهمتك هي الإجابة على السؤال باستخدام النص المعطى فقط.

قواعد إلزامية:
- استخدم فقط المعلومات الموجودة في السياق.
- لا تضف أي معلومة غير مذكورة صراحة.
- مسموح بإعادة صياغة المعلومة طالما المعنى موجود حرفيًا في السياق.
- : "إذا كان الجواب موجود في أكثر من chunk، اختر الأنسب."
- لو الإجابة غير موجودة بوضوح، قل حرفيًا: I don't know.
- حافظ على اللغة الأصلية للسؤال في الإجابة (إنجليزي أو عربي). 
- انقل الأرقام ووحداتها كما وردت حرفيًا في السياق
- لا تترجم الوحدات
- لا تختصر أو تعمم

"""

    def build_prompt(self, question: str, chunks: list):
        """
        يبني نص البرومبت من الـ chunks والسؤال
        """
        context_blocks = []
        sources = []

        for i, c in enumerate(chunks, start=1):
            if isinstance(c.get("metadata"), dict):
                source = f"{c['metadata'].get('source', 'unknown')}:page{c['metadata'].get('page', '')}"
            else:
                source = f"{c.get('metadata', 'unknown')}:page"
            sources.append(source)

            context_blocks.append(f"[{i}] {c['text']}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
السياق:
{context}

السؤال:
{question}

أجب الآن:
"""
        return prompt, sources

    def generate(self, prompt: str):
        """
        يولّد الإجابة باستخدام Llama 3.1 Instruct
        """
        logging.info("Generating answer from LLM...")
        logging.info(f"Prompt: {prompt}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        mlflow.log_param("llama_model", self.model_name)

        llm_start_time = time.time()
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                top_p=0.9,
            )
            llm_end_time = time.time()

            answer = completion.choices[0].message.get("content", "").strip()
            logging.info(f"LLM Answer: {answer}")
            if not answer:
                return "Empty answer from LLM."
            
            # Log metrics
            llm_latency_ms = (llm_end_time - llm_start_time) * 1000
            mlflow.log_metric("llama_response_length", len(answer))
            mlflow.log_metric("llm_latency_ms", llm_latency_ms)
            return answer

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return "Error generating LLM response."


@step(enable_cache=False)
def llama_llm_generation_step(
    question: str,
    chunks: list
) -> dict:
    generator = LlamaInstructLLMGenerator()

    prompt, sources = generator.build_prompt(question, chunks)
    answer = generator.generate(prompt)

    chunk_ids = [
        c.get("chunk_id") if isinstance(c, dict) else None
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunk_ids": chunk_ids,
    }
