            # api_key="",
import os
import logging
from huggingface_hub import InferenceClient
from zenml import step


class LlamaInstructLLMGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.client = InferenceClient(
            api_key=os.environ.get("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"),
        )
        self.model_name = model_name

        # تعليمات النظام
        self.system_prompt = """
أنت مساعد ذكي.
مهمتك هي الإجابة على السؤال باستخدام النص المعطى فقط.

قواعد إلزامية:
- استخدم فقط المعلومات الموجودة في السياق.
- لا تضف أي معلومة غير مذكورة صراحة.
- مسموح بإعادة صياغة المعلومة طالما المعنى موجود حرفيًا في السياق.
- لو الإجابة غير موجودة بوضوح، قل حرفيًا: لا أعلم
- أجب باللغة العربية وبإجابة مباشرة مختصرة.
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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                top_p=0.9,
            )

            answer = completion.choices[0].message.get("content", "").strip()
            if not answer:
                return "لا أعلم"
            return answer

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return "لا أعلم"


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
        "chunk_ids": chunk_ids
    }
