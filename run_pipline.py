from pipelines.pipeline import rag_pipeline
import pickle
from sentence_transformers import SentenceTransformer
import mlflow
import numpy as np
import os
from typing import Optional , List, Dict

if __name__ == "__main__":
    mlflow.set_experiment("DeepRAG-Evaluation")

    with mlflow.start_run(run_name="baseline_rag_pipeline"):
        model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        questions = "How many shares were reserved for future issuance under the Alphabet 2021 Stock Plan as of December 31, 2023?"

        doc_path = r"E:\pyDS\Buliding Rag System\data"
        query_vec = model.encode(questions, normalize_embeddings=True)
        query_vec_list = query_vec.tolist() if isinstance(query_vec, __import__('numpy').ndarray) else query_vec   # Convert numpy array to list for Pydantic serialization
        run = rag_pipeline(query=questions, query_vector=query_vec_list, filters=None, doc_path=doc_path , relevant_ids=['d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e'])



# For Testing set of questions and relevant chunk IDs
'''
Based on the provided text chunks, here are some questions that can be extracted and the relevant chunk IDs for each:

### 1. **What was the number of shares used in diluted per share computation?**
**Relevant Chunk IDs:** `1c773723bf56c456b2d7ec89ef9013f2b7f1ba8181b9c8728b660d79ec3a7d5a`, `21308bd6963b0bf22a65e11d0d144d8968072b363af0f8c625a66881c0edc552`, `d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e`
**Reason:** All three chunks contain the specific line "Number of shares used in per share computation 6,799"

### 2. **What was the diluted net income per share amount?**
**Relevant Chunk IDs:** `21308bd6963b0bf22a65e11d0d144d8968072b363af0f8c625a66881c0edc552`, `d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e`
**Reason:** Both chunks contain "Diluted net income per share $ 5.80 $ 5.80 $ 5.80"

### 3. **What is the Alphabet 2021 Stock Plan and what does it include?**
**Relevant Chunk IDs:** `21308bd6963b0bf22a65e11d0d144d8968072b363af0f8c625a66881c0edc552`, `d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e`
**Reason:** These chunks describe the Alphabet Amended and Restated 2021 Stock Plan and its provisions for RSUs

### 4. **How many shares were reserved for future issuance under the Alphabet 2021 Stock Plan as of December 31, 2023?**
**Relevant Chunk IDs:** `d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e`
**Reason:** Contains "As of December 31, 2023, there were 723 million shares of Class C stock reserved for future issuance under the Alphabet 2021 Stock Plan"

### 5. **What was the total stock-based compensation expense for 2021, 2022, and 2023?**
**Relevant Chunk IDs:** `d6c4ecb45d37505f8731ada6e883799ad7357fcdd8277afb61c9cf7dfd574e8e`
**Reason:** Contains "For the years ended December 31, 2021, 2022, and 2023, total SBC expense was $15.7 billion, $19.5 billion, and $22.1 billion"

### 6. **What was the weighted-average grant-date fair value of RSUs granted in 2021 and 2022?**
**Relevant Chunk IDs:** `4ae2f4a8574c481a0facdfd049f2c8ab302ccc2fc195153705e11ce435106898`
**Reason:** Contains "The weighted-average grant-date fair value of RSUs granted during the years ended December 31, 2021 and 2022 was $97.46 and $127.22, respectively"

### 7. **What was the unrecognized compensation cost related to unvested RSUs as of December 31, 2023, and over what period is it expected to be recognized?**
**Relevant Chunk IDs:** `464e14c8493e0acc1e8d0826ee73cf50ea219dcd386fe5303efa5a47e661a2df`
**Reason:** Contains "As of December 31, 2023, there was $33.5 billion of unrecognized compensation cost related to unvested RSUs. This amount is expected to be recognized over a weighted-average period of 2.5 years."

### 8. **What was Alphabet's effective tax rate for 2021, 2022, and 2023?**
**Relevant Chunk IDs:** `e4ed770d382bbc86a368c7708ac6cdc69c0ae7be3afd2a812feaa1dac28b2d4c`
**Reason:** Contains the reconciliation table showing effective tax rates of 16.2%, 15.9%, and 13.9% for 2021, 2022, and 2023 respectively

### 9. **What were the significant components of deferred tax assets as of December 31, 2023?**
**Relevant Chunk IDs:** `da85d932db2cac00a92d0b6e1535d5a4b06ac6d63eafb3a7dadc9f0a1e6fd806`, `868eee9296ec6cefa1ccd583d6285b5b8cc6b3bec7d75e06ca4f670fab0ab00d`, `a349b5e0772b8a33053f20cf949975117633b66dcfcdf3c067324d7d69c2cc1b`
**Reason:** These chunks contain the breakdown of deferred tax assets including accrued employee benefits, accruals and reserves, operating leases, capitalized R&D, tax credits, and net operating losses

### 10. **What was the provision for income taxes for 2021, 2022, and 2023?**
**Relevant Chunk IDs:** `96fc2fa0afef4f2c2f394ba9930e0c86af456156738c84f3b1dce5b3164f0b90`, `94f49f7c875da865d2ae13f190cb1a95e0d4783899da34b6ab2ced20a462e448`
**Reason:** Contain the table showing provision for income taxes of $14,701, $11,356, and $11,922 million for 2021, 2022, and 2023 respectively

These questions cover the main financial and accounting information present in the text chunks, with each question directly tied to specific data points found in the provided excerpts.
'''



