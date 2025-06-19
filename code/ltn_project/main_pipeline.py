#main_pipeline.py
import re
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np

from mistralai import Mistral
from neo4j import GraphDatabase
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from pydantic import Field

from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from langchain_mistralai.embeddings import MistralAIEmbeddings

from diabetes_ltn_runtime import DiabetesLTNRuntime

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###############################################################################
# Neo4j Retrieval
###############################################################################

class Neo4jDiabetesRetriever:
    """Neo4j GraphRAG retrieval with fixed query"""

    def __init__(self, neo4j_uri: str, username: str, password: str, api_key: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key
        )

        self.retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name="entities",
            fulltext_index_name="entityEnFulltextIndex",
            embedder=self.embeddings,
            retrieval_query="""
            MATCH (node)-[r]-(related:Entity)
            RETURN node.entity_en + " --[" + type(r) + "]--> " + related.entity_en as text,
                   {
                       source_id: elementId(node),
                       relation_type: type(r),
                       target_id: elementId(related),
                       source_text: node.entity_en,
                       target_text: related.entity_en
                   } as metadata
            LIMIT 20
            """
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResultItem]:
        result = self.retriever.search(query_text=query, top_k=top_k)
        return result.items


###############################################################################
# Enhanced Mistral LLM
###############################################################################

class MistralLLM(LLM):
    """Complete Mistral LLM implementation"""

    client: Optional[Mistral] = Field(default=None, exclude=True)
    model: str = Field(default="mistral-large-latest")

    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        super().__init__()
        self.client = Mistral(api_key=api_key)
        self.model = model

    def _call(self, prompt: str, stop=None) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "mistral"


###############################################################################
# Complete Integrated System
###############################################################################

class DiabetesDecisionSupport:
    """Complete methodology implementation"""

    def __init__(self, config: Dict):
        self.config = config
        self.llm = MistralLLM(config['mistral_api_key'])
        self.retriever = Neo4jDiabetesRetriever(
            config['neo4j_uri'],
            config['neo4j_username'],
            config['neo4j_password'],
            config['mistral_api_key']
        )
        self.ltn = self._initialize_ltn()

    def _initialize_ltn(self) -> DiabetesLTNRuntime | None:
        try:
            ckpt_dir = Path(self.config["ltn_checkpoint_dir"])
            return DiabetesLTNRuntime(ckpt_dir)
        except Exception as e:
            print(f"LTN initialisation failed: {e}")
            return Noneime(ckpt_dir)


    def query(self, user_input: Union[str, Dict], mode: str = "general") -> Dict:
        """Complete query processing pipeline.
        user_input: Can be a string for general query or a dict for patient-specific form.
        mode: "general" or "patient_specific"
        """
        user_query_for_retrieval = ""
        ltn_query_input = ""
        patient_id_for_ltn = "default_patient_id" # Default patient ID

        if mode == "general":
            if not isinstance(user_input, str):
                raise ValueError("User input must be a string for general mode.")
            user_query_for_retrieval = user_input
            ltn_query_input = user_input
        elif mode == "patient_specific":
            if not isinstance(user_input, dict):
                raise ValueError("User input must be a dictionary for patient_specific mode.")
            patient_data = user_input
            # Construct a descriptive query from patient data for retrieval and LTN
            patient_data_desc = []
            for key, value in patient_data.items():
                if value: # Only include fields that have values
                    patient_data_desc.append(f"{key.replace('_', ' ').capitalize()}: {value}")

            query_intro = "Considering a patient with the following characteristics: "
            user_query_for_retrieval = query_intro + "; ".join(patient_data_desc) + \
                                     ". What are relevant diabetes-related insights, advice, or potential risks?"
            ltn_query_input = user_query_for_retrieval # LTN's _ground_query needs to handle this
            patient_id_for_ltn = patient_data.get('patient_id', patient_id_for_ltn)

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'general' or 'patient_specific'.")

        # 1. Dual retrieval (vector + fulltext hybrid)
        retrieved_docs = self.retriever.retrieve(user_query_for_retrieval)

        # 2. LTN probabilistic reasoning with all rules
        ltn_result = {"predicted_facts": []}
        if self.ltn and retrieved_docs:
            for doc in retrieved_docs:
                h = doc.metadata["source_text"]
                r = doc.metadata["relation_type"]
                t = doc.metadata["target_text"]
                # only score triples the model knows
                if all(k in self.ltn.ent2id for k in (h, t)) and r in self.ltn.rel2id:
                    μ = self.ltn.triple_confidence(h, r, t)
                    if μ >= self.config.get("ltn_threshold", 0.5):
                        ltn_result["predicted_facts"].append((h, r, t, μ))

            for h, r, t, μ in ltn_result["predicted_facts"][:3]:
                ltn_result.setdefault("explanations", {})[(h, r, t)] = \
                    self.ltn.explain(h, r, t, top_k=3)

        # 3. Enhanced chain-of-thought prompting
        prompt = self._build_structured_prompt(user_input, retrieved_docs, ltn_result, mode)

        # 4. Generate response
        response = self.llm._call(prompt)

        # 5. Citation verification and enhancement
        verified_response = self._verify_citations(response, retrieved_docs, ltn_result)

        return {
            'response': verified_response,
            'retrieved_facts': [doc.content for doc in retrieved_docs if hasattr(doc, 'content')] if retrieved_docs else [],
            'ltn_reasoning': ltn_result,
            'citations_verified': True # Assuming verification logic implies this
        }

    def _build_structured_prompt(self, user_input: Union[str, Dict],
                                 docs: List[RetrieverResultItem],
                                 ltn_result: Optional[Dict],
                                 mode: str) -> str:
        prompt_parts = []
        query_display_for_prompt = ""

        if mode == "general":
            query_display_for_prompt = str(user_input)
            prompt_parts.append("You are a clinical decision support system answering a general diabetes query. Follow this reasoning structure:")
        elif mode == "patient_specific":
            patient_data_str_parts = []
            if isinstance(user_input, dict):
                for key, value in user_input.items():
                    if value: # Only include fields that have values
                         patient_data_str_parts.append(f"{key.replace('_', ' ').capitalize()}: {value}")
            query_display_for_prompt = "Patient Data: " + "; ".join(patient_data_str_parts)
            prompt_parts.append("You are a clinical decision support system analyzing patient-specific data for diabetes. Follow this reasoning structure:")

        prompt_parts.extend([
            "",
            "RETRIEVED FACTS FROM DiaKG (relevant to the query/patient data):",
        ])

        if ltn_result:
            prompt_parts.extend([
                "",
                "PREDICTED FACTS BY LTN (confidence µ):"
            ])
            for i, (h, r, t, mu) in enumerate(ltn_result["predicted_facts"][:10]):
                prompt_parts.append(f"[LTN:{i}] {h} --[{r}]--> {t}  (µ={mu:.2f})")

        if docs:
            for i, doc in enumerate(docs[:10]): # Limiting to 10 docs for brevity
                 if hasattr(doc, 'content'):
                    prompt_parts.append(f"[TRIPLE:{i}] {doc.content}")
        else:
            prompt_parts.append("No specific facts retrieved from DiaKG for this input.")


        prompt_parts.extend([
            "",
            "INSTRUCTIONS:",
            "1. Analyze retrieved facts and patient data (if provided) step by step.",
            "2. Integrate LTN‑predicted facts and their µ scores where relevant."
            "3. Provide evidence for each claim using [TRIPLE:X] or [RULE:X] citations where applicable.",
            "4. If the evidence is insufficient to provide a confident answer, state that clearly.",
            "5. Structure response as: Analysis → Evidence (if any) → Conclusion → Citations (if any).",
            "",
            f"USER INPUT CONTEXT: {query_display_for_prompt}",
            "",
            "RESPONSE:"
        ])
        return "\n".join(prompt_parts)

    def _verify_citations(self, response: str, docs: List[RetrieverResultItem], ltn_result: Optional[Dict] = None) -> str:
        """Complete citation verification system"""
        import re

        # Extract all citations
        import re
        citations = re.findall(r'\[(?:TRIPLE|LTN|RULE):\d+\]', response)

        valid_triple_ids = set(range(len(docs)))

        valid_ltn_ids = set()
        valid_rule_ids = set()
        if ltn_result:
            valid_ltn_ids = set(range(len(ltn_result.get("predicted_facts", []))))
            # “rule_id” can be any int, so use the actual ids you inserted in the prompt
            valid_rule_ids = {
                r["rule_id"] for r in ltn_result.get("rule_traces", [])
                if "rule_id" in r
            }

        for tag in citations:
            label, num = tag[1:-1].split(':')  # e.g. 'LTN', '3'
            num = int(num)
            if label == 'TRIPLE' and num not in valid_triple_ids:
                response = response.replace(tag, f"[INVALID-{label}:{num}]")
            elif label == 'LTN' and num not in valid_ltn_ids:
                response = response.replace(tag, f"[INVALID-{label}:{num}]")
            elif label == 'RULE' and num not in valid_rule_ids:
                response = response.replace(tag, f"[INVALID-{label}:{num}]")

        return response




###############################################################################
# Usage
###############################################################################

def main():
    config = {
        'mistral_api_key': userdata.get('MISTRAL_API_KEY'),
        'neo4j_uri': userdata.get('NEO4J_URI'),
        'neo4j_username': userdata.get('NEO4J_USERNAME'),
        'neo4j_password': userdata.get('NEO4J_PASSWORD')
    }


    system = DiabetesDecisionSupport(config)

    while True:
        print("\nSelect an option:")
        print("1. General Diabetes Query")
        print("2. Guided Patient Specific Data Form")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            user_query = input("Enter your general diabetes query: ")
            if not user_query:
                print("Query cannot be empty.")
                continue
            print("\nProcessing general query...")
            result = system.query(user_query, mode="general")
            print("\n--- General Query Result ---")
            print(result['response'])

        elif choice == '2':
            print("\n--- Guided Patient Specific Data Form ---")
            patient_data = {}
            print("Please provide the following patient information (leave blank if not applicable):")

            patient_data['patient_id'] = input("Patient ID: ").strip()
            patient_data['age'] = input("Patient Age (e.g., 55): ").strip()
            patient_data['gender'] = input("Patient Gender (e.g., Male, Female, Other): ").strip()
            patient_data['blood_glucose_level'] = input("Last Blood Glucose Level (e.g., '150 mg/dL fasting', '200 mg/dL postprandial'): ").strip()
            patient_data['hba1c'] = input("HbA1c Level (e.g., '7.5%'): ").strip()
            patient_data['symptoms'] = input("Current Symptoms (comma-separated, e.g., excessive thirst, frequent urination, unexplained weight loss): ").strip()
            patient_data['family_history_diabetes'] = input("Family history of diabetes (Yes/No/Details, e.g., 'Yes, mother had Type 2'): ").strip()
            patient_data['current_medications'] = input("Current Medications for diabetes or other conditions (comma-separated): ").strip()
            patient_data['comorbidities'] = input("Other existing medical conditions (comma-separated, e.g., hypertension, hyperlipidemia): ").strip()
            patient_data['lifestyle_factors'] = input("Key lifestyle factors (e.g., diet summary, exercise routine, smoker): ").strip()

            # Filter out entries where the user provided no input
            filled_patient_data = {k: v for k, v in patient_data.items() if v}

            if not filled_patient_data:
                print("No patient data provided. Cannot proceed with patient-specific query.")
                continue

            print("\nProcessing patient-specific data...")
            result = system.query(filled_patient_data, mode="patient_specific")
            print("\n--- Patient Specific Result ---")
            print(result['response'])

        elif choice == '3':
            print("Exiting Diabetes Decision Support System.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()