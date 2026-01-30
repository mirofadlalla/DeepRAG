"""
Query Expansion والـ Reformulation:
1. إنشاء متغيرات من الاستعلام (paraphrasing)
2. إضافة مرادفات
3. توسيع الاستعلام بكلمات مهمة
4. Multi-hop question decomposition
"""

from zenml import step
from typing import List, Dict
import logging
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="E:\pyDS\Buliding Rag System\.env")

class QueryExpander:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Query expander using LLM
        """
        self.client = InferenceClient(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )
        self.model_name = model_name
        
        logging.info("QueryExpander initialized")

    def _generate_paraphrases(self, query: str, num_paraphrases: int = 3) -> List[str]:
        """
        توليد إعادة صياغة للاستعلام الأصلي
        """
        system_prompt = """أنت متخصص في إعادة صياغة الأسئلة.
        قم بإعادة صياغة السؤال بطرق مختلفة تحافظ على نفس المعنى الأساسي.
        اخرج فقط الأسئلة المعاد صياغتها، سطر واحد لكل سؤال."""
        
        prompt = f"""أعد صياغة السؤال التالي بـ {num_paraphrases} طرق مختلفة:
        السؤال: {query}
        
        الأسئلة المعاد صياغتها:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            
            response = completion.choices[0].message.get("content", "").strip()
            paraphrases = [line.strip() for line in response.split("\n") if line.strip()]
            return paraphrases[:num_paraphrases]
        except Exception as e:
            logging.error(f"Error generating paraphrases: {e}")
            return [query]  # Fallback to original query

    def _extract_keywords(self, query: str) -> List[str]:
        """
        استخراج الكلمات المفتاحية من الاستعلام
        """
        # Common stop words (can be expanded)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "is", "are", "was", "were", "be", "been", "being",
            "كل", "في", "على", "من", "هو", "هي", "هم", "أن"
        }
        
        # Split the query and filter stop words
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords

    def _decompose_question(self, query: str) -> List[str]:
        """
        تحليل السؤال إلى أسئلة فرعية (Multi-hop decomposition)
        """
        system_prompt = """أنت متخصص في تحليل الأسئلة المعقدة.
        قم بتقسيم السؤال إلى أسئلة فرعية أبسط.
        اخرج كل سؤال فرعي في سطر منفصل."""
        
        prompt = f"""حلل السؤال التالي إلى أسئلة فرعية:
        السؤال: {query}
        
        الأسئلة الفرعية:"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5,
            )
            
            response = completion.choices[0].message.get("content", "").strip()
            subquestions = [
                line.strip() 
                for line in response.split("\n") 
                if line.strip() and not line.startswith("-")
            ]
            return subquestions
        except Exception as e:
            logging.error(f"Error decomposing question: {e}")
            return [query]  # Fallback

    def expand(self, query: str, include_paraphrases: bool = True, 
              include_decomposition: bool = False) -> Dict[str, List[str]]:
        """
        توسيع الاستعلام الشامل
        
        Returns:
            {
                "original": str,
                "paraphrases": List[str],
                "keywords": List[str],
                "subquestions": List[str],
                "all_queries": List[str]
            }
        """
        result = {
            "original": query,
            "paraphrases": [],
            "keywords": self._extract_keywords(query),
            "subquestions": [],
        }
        
        if include_paraphrases:
            result["paraphrases"] = self._generate_paraphrases(query, num_paraphrases=2)
        
        if include_decomposition:
            result["subquestions"] = self._decompose_question(query)
        
        # Aggregate all queries
        all_queries = [query]
        all_queries.extend(result["paraphrases"])
        all_queries.extend(result["subquestions"])
        
        result["all_queries"] = list(set(all_queries))  # Remove duplicates
        
        logging.info(f"Expanded query from 1 to {len(result['all_queries'])} queries")
        
        return result


@step(enable_cache=False)
def query_expansion_step(
    query: str,
    include_paraphrases: bool = True,
    include_decomposition: bool = False,
) -> Dict[str, List[str]]:
    """
    خطوة توسيع الاستعلام
    """
    expander = QueryExpander()
    expanded = expander.expand(
        query,
        include_paraphrases=include_paraphrases,
        include_decomposition=include_decomposition
    )
    return expanded
