from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

from zenml import step

class QwenLLMGenerator4bit:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        ).to(self.device)

        self.system_prompt = """
مهمتك هي الإجابة على السؤال باستخدام النص المعطى فقط.
 قواعد إلزامية:
- استخدم فقط المعلومات الموجودة في السياق.
- ممنوع التخمين أو الاستنتاج.
- لو الإجابة غير موجودة بوضوح، قل حرفيًا: "لا أعلم".
- أجب باللغة العربية وبإجابة مباشرة.
"""

    def build_prompt(self, question: str, chunks: list):
        context_blocks = []
        sources = []

        for i, c in enumerate(chunks, start=1):
            source = f"{c['metadata']['source']}:page{c['metadata'].get('page', '')}"
            sources.append(source)
            context_blocks.append(f"[{i}] المصدر: {source}\n{c['text']}")

        context = "\n\n".join(context_blocks)
        prompt = f"""
السياق:
{context}

السؤال:
{question}

الإجابة:
"""
        return prompt, sources

    def generate(self, prompt: str):
        full_prompt = f"### System:\n{self.system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("### Assistant:")[-1].strip()


@step()
def qwen_llm_generation_step(
    question: str,
    chunks: list
) -> str:
    generator = QwenLLMGenerator4bit()
    prompt, sources = generator.build_prompt(question, chunks)
    answer = generator.generate(prompt)
    return {"answer" : answer, "sources": sources}